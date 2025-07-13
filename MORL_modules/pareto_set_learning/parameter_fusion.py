

def soft_update_mlp(
        source_mlp: nn.Sequential,
        target_mlp: nn.Sequential,
        alpha: float,
        phi: float = None  # Note: you mentioned phi but didn't specify its use
) -> None:
    """
    Perform soft update of target network parameters using source network.

    For each parameter p2 in target_mlp and corresponding p1 in source_mlp:
    p2 = alpha * p1 + (1 - alpha) * p2

    Args:
        source_mlp: Source MLP network (e.g., main network)
        target_mlp: Target MLP network (e.g., target network)
        alpha: Interpolation factor (0 = no update, 1 = full copy)
        phi: Additional parameter (currently unused, kept for interface compatibility)
    """
    # Assert networks have same architecture
    assert_same_architecture(source_mlp, target_mlp)

    # Perform soft update
    with torch.no_grad():
        for source_param, target_param in zip(source_mlp.parameters(), target_mlp.parameters()):
            target_param.data.copy_(
                alpha * source_param.data + (1.0 - alpha) * target_param.data
            )


def assert_same_architecture(mlp1: nn.Sequential, mlp2: nn.Sequential) -> None:
    """
    Assert that two MLPs have the same architecture.

    Args:
        mlp1: First MLP
        mlp2: Second MLP

    Raises:
        AssertionError: If architectures don't match
    """
    # Check same number of modules
    assert len(mlp1) == len(mlp2), f"Different number of modules: {len(mlp1)} vs {len(mlp2)}"

    # Check each module type and parameters
    for i, (module1, module2) in enumerate(zip(mlp1, mlp2)):
        # Check module types
        assert type(module1) == type(module2), \
            f"Module {i}: Different types {type(module1)} vs {type(module2)}"

        # Check parameter shapes for layers with parameters
        if hasattr(module1, 'weight') and hasattr(module2, 'weight'):
            assert module1.weight.shape == module2.weight.shape, \
                f"Module {i}: Different weight shapes {module1.weight.shape} vs {module2.weight.shape}"

        if hasattr(module1, 'bias') and hasattr(module2, 'bias'):
            if module1.bias is not None and module2.bias is not None:
                assert module1.bias.shape == module2.bias.shape, \
                    f"Module {i}: Different bias shapes {module1.bias.shape} vs {module2.bias.shape}"
            elif module1.bias is not None or module2.bias is not None:
                assert False, f"Module {i}: One has bias, other doesn't"


# Alternative version with more detailed architecture checking
def assert_same_architecture_detailed(mlp1: nn.Sequential, mlp2: nn.Sequential) -> None:
    """More detailed architecture comparison"""

    def get_architecture_signature(mlp):
        """Extract architecture signature from MLP"""
        signature = []
        for module in mlp:
            if isinstance(module, nn.Linear):
                signature.append(('Linear', module.in_features, module.out_features, module.bias is not None))
            elif isinstance(module, nn.Dropout):
                signature.append(('Dropout', module.p))
            elif isinstance(module, nn.LayerNorm):
                signature.append(('LayerNorm', tuple(module.normalized_shape)))
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                signature.append((type(module).__name__,))
            else:
                signature.append((type(module).__name__, 'unknown'))
        return signature

    sig1 = get_architecture_signature(mlp1)
    sig2 = get_architecture_signature(mlp2)

    assert sig1 == sig2, f"Architecture mismatch:\nMLP1: {sig1}\nMLP2: {sig2}"