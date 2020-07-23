import setuptools

setuptools.setup(
    name="lokki", 
    version="0.0.2",
    author="Michael Greer",
    author_email="michael.j.greer1@gmail.com",
    description="Lokki: an automatic machine learning framework for 16s metagenomic data",
    long_description="Lokki: an automatic machine learning framework for 16s metagenomic data",
    long_description_content_type="text/markdown",
    license='Apache 2.0',
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.16.4',
        'pandas==0.25.0',
        'treelib==1.6.1',
        'matplotlib==3.2.2',
        'dill==0.3.2',
        'scikit-learn==0.20.3',
        'scikit-optimize==0.7.4'
    ],
    url="https://github.com/greermj/Lokki",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
