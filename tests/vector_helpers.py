"""Scalar-shaped driver for testing the native batch without a second executor."""
from motion_challenge.simulation.vector_physics import VectorPhysics
from dataclasses import astuple

class VectorWorld:
    def __init__(self, profile):
        self.pool = VectorPhysics(profile,1,1)
        self.world = self.pool.world
    def __getattr__(self,name): return getattr(self.world,name)
    def reset(self,motion):
        self.world = self.pool.reference_world(motion,64)
        self.pool.reset([0],[self.world])
        return self.state()
    def advance(self,command,target): self.pool.advance([0],[astuple(command)],[target])
    def state(self): return self.pool.states([0])[0]
