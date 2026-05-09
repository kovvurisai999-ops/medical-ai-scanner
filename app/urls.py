from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('user_login/', views.user_login_view, name='user_login'),
    path('admin_login/', views.admin_login_view, name='admin_login'),
    path('signup/', views.signup_view, name='signup'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('doctor_dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('hospital_dashboard/', views.hospital_dashboard, name='hospital_dashboard'),
    path('doctor_settings/', views.doctor_settings, name='doctor_settings'),
    path('approve/<int:id>/', views.approve_report, name='approve_report'),
    path('reject/<int:id>/', views.reject_report, name='reject_report'),
    path('delete/<int:id>/', views.delete_report, name='delete_report'),
    path('upload/', views.upload, name='upload'),
    path('patient_report_login/', views.patient_report_login, name='patient_report_login'),
    path('patient_report_download/', views.patient_report_download, name='patient_report_download'),
    path('logout/', views.logout_view, name='logout'),

]
