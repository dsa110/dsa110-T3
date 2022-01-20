from setuptools import setup
from dsautils.version import get_git_version

setup(name='dsa110-T3',
      version=get_git_version(),
      url='http://github.com/dsa110/dsa110-T3/',
      author='Dana Simard',
      author_email='dana.simard@astro.caltech.edu',
      packages=['dsaT3'],
      package_data={'dsaT3':['data/*']},
      install_requires=['astropy',
                        'casatools',
                        'casatasks',
                        'casadata',
                        'matplotlib',
                        'numpy==1.19.5',
                        'pytest',
                        'codecov',
                        'coverage',
                        'pyyaml',
                        'scipy',
                        'etcd3',
                        'structlog',
                        'dsa110-pyutils',
                        'dsa110-meridian-fs',
                        'dsa110-calib',
                        'sigpyproc',
                        'slack',
                        'slackclient',
                        'tensorflow==2.5.0',
                        'psrqpy'
      ],
      dependency_links = [
          "https://github.com/dsa110/dsa110-antpos/tarball/master#egg=dsa110-antpos",
          "https://github.com/dsa110/dsa110-pyutils/tarball/master#egg=dsa110-pyutils",
          "https://github.com/dsa110/dsa110-meridian-fs/tarball/main#egg=dsa110-meridian-fs",
          "https://github.com/dsa110/dsa110-calib/tarball/main#egg=dsa110-calib",
      ]
)
