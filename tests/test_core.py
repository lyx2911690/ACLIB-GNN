"""Scientific invariants: gradients, leakage boundaries and deterministic pooling."""
import unittest
import numpy as np
import torch
from aclib.data import neighborhoods, pool, pooling_index
from aclib.metrics import classification_metrics
from aclib.model import ACLIBGNN, adversarial_embeddings, objective


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ACLIBGNN(3, 4, 2, heads=2)
        self.edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])

    def test_center_is_retained_and_isolated_node_is_finite(self):
        groups = neighborhoods(self.edges, 5, max_nodes=1)
        self.assertEqual(groups, [[0], [1], [2], [3], [4]])
        full = neighborhoods(self.edges, 5, max_nodes=0)
        embeddings = torch.arange(10, dtype=torch.float32).reshape(5, 2).requires_grad_()
        pooled = pool(embeddings, pooling_index(full, "cpu"))
        self.assertTrue(torch.equal(pooled[4], embeddings[4]))
        torch.testing.assert_close(pooled[0], embeddings[:2].mean(0))
        pooled.sum().backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_fgsm_ignores_held_out_labels_and_keeps_parameter_grads_empty(self):
        pooled = torch.randn(5, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 0])
        train = torch.tensor([0, 1])
        attacked = adversarial_embeddings(self.model, pooled, labels, train, .1)
        changed = labels.clone()
        changed[2:] = 1 - changed[2:]
        torch.testing.assert_close(attacked, adversarial_embeddings(self.model, pooled, changed, train, .1))
        torch.testing.assert_close(attacked[2:], pooled[2:])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertLessEqual(float((attacked - pooled).abs().max().detach()), .100001)

    def test_equations_and_single_backward(self):
        embedding = self.model(torch.randn(5, 3), self.edges)
        pooled = pool(embedding, pooling_index(neighborhoods(self.edges, 5), "cpu"))
        labels, train = torch.tensor([0, 1, 0, 1, 0]), torch.tensor([0, 1])
        attacked = adversarial_embeddings(self.model, pooled, labels, train, .1)
        loss, parts = objective(self.model, embedding, pooled, attacked, labels, train, .4, .6, .1)
        expected = parts["node"] + .4 * parts["original"] + .6 * parts["adversarial"] + parts["diversity"]
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertIsNotNone(self.model.conv1.lin.weight.grad)

    def test_metrics_binary_and_missing_class(self):
        metrics = classification_metrics(np.array([0, 1]), np.array([[.9, .1], [.1, .9]]))
        self.assertEqual(metrics["AUC"], 1.)
        missing = classification_metrics(np.array([0, 0]), np.array([[.9, .1], [.8, .2]]))
        self.assertIsNone(missing["AUC"])


if __name__ == "__main__":
    unittest.main()
