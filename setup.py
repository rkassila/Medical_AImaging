from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='aimaging',
      version="0.0.2",
      description="AIMaging (api_pred)",
      author="AIMaging team @LeWagon Tokyo",
      install_requires=requirements,
      packages=find_packages(),
      zip_safe=False,
      entry_points={
        'console_scripts': [
            'run-api=aimaging.api.fast:main',
            'run-streamlit=aimaging.interface.main:main',
        ]
        }
      )
