from django.contrib import admin
from django.urls import path, include
from . import views 
from django.conf import settings
from django.conf.urls.static import static
import debug_toolbar

urlpatterns = [
    path('', views.index, name = 'index'),
    path('signup', views.signup, name='signup'),
    path('signout', views.signout, name='signout'),
    path('signin', views.signin, name='signin'),
    path('predict/', views.predict, name='predict'),
    path("debug/", include("debug_toolbar.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)



