from distutils.core import setup

setup(
    name='latent Bayesian melding',
    version='0.1.1',
    author='Mingjun Zhong',
    author_email='mingjun.zhong@gmail.com',
    packages=['LatentBayesianMelding'],
    scripts=[],
    url='https://github.com/MingjunZhong/LatentBayesianMelding',
    license='',
    description='Latent Bayesian Melding for Non Intrusive Load Monitoring',
    install_requires=[
        'numpy>=1.7', 'pandas>=0.12', 'matplotlib>=1.3', 'scikit-learn>=0.13', 'scipy>=0.13'

    ],
)
