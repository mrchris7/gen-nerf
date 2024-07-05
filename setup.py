from setuptools import find_packages, setup


setup(
   name='gen-nerf',
   version='1.0',
   description='Generalizable Neural Feature Fields',
   long_description='Learning scene-level generalizable neural feature fields using NeRFs and feature distillation from pre-trained Vision Language Models, creating a unified scene representation that captures geometric and semantic properties.',
   author='Christian Maurer',
   packages=find_packages(include=['src', 'src.*'])
   #install_requires=['bar'],
)