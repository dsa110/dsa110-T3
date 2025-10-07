from setuptools import setup
from dsautils.version import get_git_version

setup(name='dsa110-T3',
      version=get_git_version(),
      url='http://github.com/dsa110/dsa110-T3/',
      author='DSA-110 Team',
      author_email='',
      packages=['dsaT3'],
      package_data={'dsaT3':['data/*']},
      install_requires=['astropy',
                        'casatools',
                        'casatasks',
                        'casadata',
                        'matplotlib',
#                        'numpy==1.26.4',  # for py310?
                        'numpy<2',
                        'pytest',
                        'codecov',
                        'coverage',
                        'pyyaml',
                        'scipy',
                        'etcd3',
                        'structlog',
                        'dsa110-pyutils',
                        'sigpyproc',
                        'slack',
                        'slackclient',
                        'tensorflow==2.11.0',
                        'psrqpy'
      ],
)
