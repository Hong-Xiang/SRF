from setuptools import setup, find_packages

setup(name='SRF',
      version='0.2.2',
      description='Scalable Reconstruction Framework.',
      url='https://github.com/tech-pi/SRF',
      author='Hong Xiang',
      author_email='hx.hongxiang@gmail.com',
      license='Apache',
      packages=find_packages('src/python'),
      package_dir={'': 'src/python'},
      install_requires=[
          'dxl-learn==0.2.1',
          'dxl-core==0.1.7',
          'doufo==0.0.4',
          'dxl-shape==0.1.2',
          'jfs==0.1.3',
          'scipy',
          'matplotlib',
          'typing',
          'h5py',
          'click',
          'jinja2',
          'pathlib',
          'numpy',
          'tqdm',
          'tensorflow'
      ],
      entry_points="""
            [console_scripts]
            srf=srf.api.cli.main:srf
      """,
      zip_safe=False)
