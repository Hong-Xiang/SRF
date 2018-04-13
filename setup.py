from setuptools import setup, find_packages
setup(name='SRF',
      version='0.0.0',
      description='Machine learn library.',
      url='https://github.com/Hong-Xiang/SRF',
      author='Hong Xiang',
      author_email='hx.hongxiang@gmail.com',
      license='MIT',
      packages=['SRF'],
      package_dir={'': 'src/python'},
      install_requires=['dxl-fs', 'click', 'dxl-shape','dxl-learn'],
      zip_safe=False)
