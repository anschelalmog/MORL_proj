# agents/mosac/policies.py
class MOSACPolicy(SACPolicy):
    def make_critic(self, features_extractor):
        pass # Create multi-objective critic

    def make_actor(self, features_extractor):
        pass  # Create actor (same as SAC)