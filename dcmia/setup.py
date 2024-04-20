import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="DCMIA",
    version="1.0.0",
    author="Javier M. Madruga and Hugo G. Parente",
    description="House detection in high resolution aerial images.",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements
)
