import os

# dirname removes the last segment of a path.
# abspath(__FILE__) returns the absolute path of a file
# So basedir is the parent directory of the directory
# this settings.py file is in.
# So this is ../machine_learner/machine_learner/settings.py
# so basedir = ../machine_learner
# this is the directory manage.py resides in
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = 's7og$shd9^4s4ji(ju1g3toy79&snz440^x1ezzs@9om2-yqb%'

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.contenttypes'
]

MIDDLEWARE = [
    'django.middleware.csrf.CsrfViewMiddleware'
]

ROOT_URLCONF = 'machine_learner.urls'

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'CET'

USE_I18N = True

USE_L10N = True

USE_TZ = True

STATIC_URL = '/static/'
