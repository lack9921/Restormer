"""Gradient accumulation wrapper with independent internal counter.

Usage:
    accum = GradientAccumWrapper(optimizer, steps=8)

    for data in loader:
        accum.zero_grad()          # no-op until it's time to zero
        loss = model(data)
        accum.backward(loss)       # auto-scale by 1/steps
        if accum.should_step():
            accum.step()           # optimizer step
            accum.reset()          # (optional; step() calls reset internally)

Key: internal _step counter is fully decoupled from the training loop's
iteration counter, avoiding off-by-one bugs.
"""
import math


class GradientAccumWrapper:
    """Wraps a single PyTorch optimizer with gradient accumulation.

    The internal ``_step`` counter tracks micro-batches, NOT training
    iterations.  ``zero_grad()`` only fires on the first micro-batch of
    each window, and ``step()`` only on the last.

    Args:
        optimizer: A ``torch.optim.Optimizer`` instance.
        accum_steps: Number of micro-batches per effective weight update.
            ``1`` disables accumulation (passthrough).
    """

    def __init__(self, optimizer, accum_steps=1):
        self.optimizer = optimizer
        self._accum_steps = max(1, int(accum_steps))
        self._step = 0   # micro-batch counter, NOT training iteration
        self._enabled = self._accum_steps > 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def zero_grad(self):
        """Zero gradients only on the **first** micro-batch of a window."""
        if not self._enabled or self._step == 0:
            self.optimizer.zero_grad()

    def backward(self, loss):
        """Backward with automatic loss scaling by ``1/accum_steps``.

        Returns:
            ``True`` if the optimizer should step after this call.
        """
        if self._enabled:
            loss = loss / self._accum_steps
        loss.backward()
        self._step += 1
        return self.should_step()

    def should_step(self):
        """Whether the current micro-batch completes an accumulation window."""
        return self._enabled and self._step % self._accum_steps == 0

    def step(self):
        """Take an optimizer step **and** reset the internal counter."""
        self.optimizer.step()
        self._step = 0

    @property
    def accum_steps(self):
        return self._accum_steps

    def state_dict(self):
        return {
            'step': self._step,
            'accum_steps': self._accum_steps,
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, sd):
        self._step = sd.get('step', 0)
        self._accum_steps = max(1, int(sd.get('accum_steps', 1)))
        self._enabled = self._accum_steps > 1
        self.optimizer.load_state_dict(sd['optimizer'])
