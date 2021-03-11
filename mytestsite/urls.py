"""mytestsite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# 원래 있던 데
# from django.contrib import admin
# from django.urls import path

# 두 번째 시도
from rest_framework import routers
# from newapp import views
from bg_removal import views
from django.conf.urls import url, include

#from here
from rest_framework import routers
from django.conf.urls import url, include

# FOR IMAGE IMPLEMENTATION AND + static url 부분
from django.conf import settings
from django.conf.urls.static import static

router = routers.DefaultRouter()
router.register(r'posts', views.PostViewset)
router.register(r'accounts', views.AccountsViewset)

urlpatterns = [
    # path('admin/', admin.site.urls),
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
