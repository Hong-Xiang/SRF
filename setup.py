from setuptools import setup, find_packages
setup(name='SRF',
      version='0.0.1',
      description='Scalable Reconstruction Framework.',
      url='https://github.com/Hong-Xiang/SRF',
      author='Hong Xiang',
      author_email='hx.hongxiang@gmail.com',
      license='MIT',
      packages=['srf'],
      package_dir={'': 'src/python'},
      install_requires=['dxl-learn'],
      zip_safe=False)
