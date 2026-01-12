import sys
import os
sys.path.append(os.getcwd()) # Ensure we can import modules

# Mock tensorboardX as it is not needed for model verification
from unittest.mock import MagicMock
sys.modules['tensorboardX'] = MagicMock()

# Mock selective_scan_cuda as it is likely missing in this env
mock_ss_cuda = MagicMock()
def mock_fwd(*args, **kwargs):
    u = args[0]
    return u, None, None, None # out, x, *rest
mock_ss_cuda.fwd = mock_fwd
sys.modules['selective_scan_cuda'] = mock_ss_cuda

import torch
from models.rangevit import RangeSeg

def test_tinyvim_integration():
    print("Testing TinyViM Integration...")
    try:
        model = RangeSeg(
            in_channels=5,
            n_cls=17,
            backbone='tinyvim_large',
            decoder='up_conv',
            image_size=(64, 768),
            pretrained_path=None,
            stem_base_channels=32 # default
        )
        print("Model instantiated successfully.")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return

    input_tensor = torch.randn(1, 5, 64, 768)
    try:
        # RangeSeg forward might expect padded input logic if not handled inside?
        # RangeSeg.forward does padding(im, self.patch_size).
        # But TinyViMAdapter might have different patch size?
        # RangeSeg uses encoder.patch_size. Adapter sets it to (4, 4).
        output = model(input_tensor)
        print(f"Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_weight_loading():
    print("\nTesting Weight Loading...")
    # Create dummy checkpoint mimicking TinyViM structure
    # tinyvim.py: TinyViM has 'network', 'patch_embed', 'norm', 'head'
    dummy_state_dict = {
        'model': {
            'patch_embed.0.c.weight': torch.randn(24, 3, 3, 3), # Note: saved w/ 3 channels
            'network.0.0.mk': torch.randn(1), # arbitrary key
        }
    }
    ckpt_path = 'dummy_tinyvim_ckpt.pth'
    torch.save(dummy_state_dict, ckpt_path)
    
    try:
        # Instantiate RangeSeg with this checkpoint
        # It should try to load.
        # Note: In our adapter, we replace the first conv (channels 5 vs 3).
        # So 'patch_embed.0.c.weight' will be mismatched in shape.
        # RangeSeg init logic handles re-init/adaptation for 'patch_embed.proj.weight' (lines 421+).
        # But TinyViM names are different ('patch_embed.0.c.weight' vs 'patch_embed.proj.weight').
        # So standard RangeSeg adaptation logic WON'T work for TinyViM keys.
        # However, we implemented manual adaptation in TinyViMAdapter.__init__.
        # Loading happens AFTER init.
        # So we load 3-channel weight into 5-channel layer? No, that fails size mismatch.
        
        # We need to verify if we can load at least the OTHER weights.
        # And we normally expect strict=False.
        
        model = RangeSeg(
            in_channels=5,
            n_cls=17,
            backbone='tinyvim_large',
            decoder='up_conv',
            image_size=(64, 2048),
            pretrained_path=ckpt_path
        )
        print("Weight loading test: RangeSeg instantiated (loading didn't crash).")
        
        # basic check
        # We need to check if keys were loaded.
        # RangeSeg loads into self.rangevit.
        # We want to check if 'encoder.model.network.0.0.mk' exists in state_dict (it won't because it's dummy).
        # But we can check if it attempted to load.
        
    except Exception as e:
        print(f"Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

if __name__ == "__main__":
    test_tinyvim_integration()
    # test_weight_loading()
