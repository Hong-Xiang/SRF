from setuptools import setup, find_packages
setup(name='SRF',
      version='0.2.1',
      description='Scalable Reconstruction Framework.',
      url='https://github.com/tech-pi/SRF',
      author='Hong Xiang',
      author_email='hx.hongxiang@gmail.com',
      license='Apache',
      packages=find_packages('src/python'),
      package_dir={'': 'src/python'},
      install_requires=['dxl-learn>=0.2.0',
                        'dxl-core>=0.1.6',
                        'tensorflow',
                        'doufo>=0.0.3',
                        'dxl-shape>=0.1.1'],
      entry_points="""
            [console_scripts]
            srf=srf.api.cli.main:srf
      """,
      zip_safe=False)
