from setuptools import setup, find_packages

setup(
    name="cad_face_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "open3d>=0.13.0",
        "trimesh>=3.9.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.5b0",
            "isort>=5.8.0",
            "flake8>=3.9.0",
        ],
        "visualization": [
            "pyvista>=0.32.0",
            "Pillow>=8.2.0",
        ],
    },
    author="Riya Sudrik",
    author_email="riyasudrik03@gmail.com",
    description="CAD Face Label Prediction using Graph Neural Networks",
    keywords="cad, machine learning, graph neural networks",
    python_requires=">=3.8",
)