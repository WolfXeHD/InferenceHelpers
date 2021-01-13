from setuptools import setup, find_packages

setup(
    name = "InferenceHelpers",
    version = "1.0",
    description = "Helper Functions for Inference",
    packages=find_packages(),
    package_data = {"InferenceHelpers": ["helping_data/*"]}
)

