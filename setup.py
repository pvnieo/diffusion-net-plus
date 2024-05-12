from setuptools import setup, find_packages

setup(
    name='diffusion_net',
    version='0.1.0',
    description='DiffusionNet implementation in PyTorch Geometric.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='My Name',
    author_email='my.email@example.com',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here as strings, e.g.,
        # 'numpy>=1.19.2',
        # 'pandas>=1.1.3',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
