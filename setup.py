from setuptools import setup, find_packages
setup(name='SRF',
      version='0.2.0',
      description='Scalable Reconstruction Framework.',
      url='https://github.com/tech-pi/SRF',
      author='Hong Xiang',
      author_email='hx.hongxiang@gmail.com',
      license='Apache',
      packages=find_packages('src/python'),
      package_dir={'': 'src/python'},
      install_requires=['dxl-learn>0.0.13',
                        'dxl-core',
                        'tensorflow',
                        'doufo'],
      entry_points="""
            [console_scripts]
            srf=srf.api.cli.main:srf
      """,
      zip_safe=False)
