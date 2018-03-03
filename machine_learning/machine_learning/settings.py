import os

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

ROOT_URLCONF = 'machine_learning.urls'

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

STATIC_URL = '/static/'
