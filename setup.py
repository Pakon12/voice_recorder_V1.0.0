from setuptools import setup, find_packages 
 
setup( 
    name="voice-recorder", 
    version="1.0.0", 
    packages=find_packages(), 
    install_requires=[ 
        "pyaudio>=0.2.11", 
        "numpy>=1.19.0", 
        "scipy>=1.5.0", 
        "matplotlib>=3.3.0", 
        "keyboard>=0.13.5" 
    ], 
    author="Your Name", 
    author_email="your.email@example.com", 
    description="A high-quality audio recorder with voice detection", 
    long_description=open("README.md").read(), 
    long_description_content_type="text/markdown", 
    url="https://github.com/yourusername/voice-recorder", 
    classifiers=[ 
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent", 
    ], 
    python_requires=">=3.7", 
) 
