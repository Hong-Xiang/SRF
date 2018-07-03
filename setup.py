from setuptools import setup, find_packages
setup(name='SRF',
      version='0.0.18',
      description='Scalable Reconstruction Framework.',
      url='https://github.com/Hong-Xiang/SRF',
      author='Hong Xiang',
      author_email='hx.hongxiang@gmail.com',
      license='MIT',
      packages=find_packages('src/python'),
      package_dir={'': 'src/python'},
      install_requires=['dxl-learn>0.0.13', 'dxl-core', 'dxl-data>=0.0.17'],
      entry_points="""
            [console_scripts]
            srf=srf.cli.main:srf
      """,
      zip_safe=False)
