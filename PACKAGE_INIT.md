# Package init files

If you use the recommended folder layout, create empty __init__.py files in:
- Network/__init__.py
- loss/__init__.py
- evaluation/__init__.py
- scripts/__init__.py (optional)

This makes `from Network.Unet3D import UNet3D` style imports work reliably.
