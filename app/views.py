from django.shortcuts import render, redirect, get_object_or_404
from .ml_model import predict_image
from .utils import generate_pdf
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from .models import Scan, UserProfile, Hospital
from django.utils import timezone
from django.db import models

def home(request):
    return render(request, "home.html")

from django.contrib.auth.models import User

def user_login_view(request):
    if request.method == "POST":
        request.session['role'] = 'user'
        username = request.POST.get('username', '').strip().lower()
        full_name = request.POST.get('full_name', '').strip()
        request.session['username'] = username
        request.session['full_name'] = full_name
        
        user, _ = User.objects.get_or_create(username=username)
        user.first_name = full_name[:30] # safe truncation
        user.save()
        
        return redirect('user_dashboard')
    return render(request, "login.html", {"role": "user"})

def admin_login_view(request):
    if request.method == "POST":
        if request.POST.get('username') == 'reddy' and request.POST.get('password') == 'Sai@2431':
            request.session['role'] = 'admin'
            return redirect('doctor_dashboard')
        return render(request, "login.html", {"role": "admin", "error": "Invalid Username or Password!"})
    return render(request, "login.html", {"role": "admin"})

def signup_view(request):
    if request.method == "POST":
        request.session['role'] = 'user'
        username = request.POST.get('username', '').strip().lower()
        full_name = request.POST.get('full_name', '').strip()
        request.session['username'] = username
        request.session['full_name'] = full_name
        
        user, _ = User.objects.get_or_create(username=username)
        user.first_name = full_name[:30]
        user.save()
        
        return redirect('user_dashboard')
    return render(request, "login.html", {"role": "signup"})

def logout_view(request):
    request.session.flush()
    return redirect('home')

def user_dashboard(request):
    if request.session.get('role') != 'user':
        return redirect('user_login')
    return render(request, "user_dashboard.html")

def doctor_dashboard(request):
    if request.session.get('role') != 'admin':
        return redirect('admin_login')
        
    scans = Scan.objects.all().order_by('-created_at')
    search_query = request.GET.get('search', '').strip()
    
    if search_query:
        from django.db.models import Q
        scans = scans.filter(Q(user__first_name__icontains=search_query) | Q(user__username__icontains=search_query))
    
    # Advanced SaaS Analytics
    stats = {
        "total": scans.count(),
        "pending": scans.filter(status='AI_COMPLETED').count(),
        "approved": scans.filter(status='APPROVED').count(),
        "critical": scans.filter(is_critical=True).count(),
        "brain": scans.filter(scan_type='brain').count(),
        "bone": scans.filter(scan_type='bone').count(),
        "chest": scans.filter(scan_type='chest').count(),
    }
    
    return render(request, "doctor_dashboard.html", {
        "patients": scans, # Keeping template variable name for compatibility
        "stats": stats
    })

def hospital_dashboard(request):
    if request.session.get('role') != 'admin':
        return redirect('admin_login')
        
    scans = Scan.objects.all()
    today = timezone.now().date()
    scans_today = scans.filter(created_at__date=today)
    
    # Calculate Business Metrics (Startup SaaS)
    total_count = scans.count()
    revenue_per_scan = 50 # Fixed value for demo
    total_rev = total_count * revenue_per_scan
    today_rev = scans_today.count() * revenue_per_scan
    
    critical_scans = scans.filter(is_critical=True).order_by('-created_at')
    pending_reviews = scans.filter(status='AI_COMPLETED').count()
    approved_count = scans.filter(status='APPROVED').count()
    
    # AI Performance Analytics
    avg_confidence = scans.aggregate(models.Avg('confidence'))['confidence__avg'] or 0
    
    # Modality Distribution (Actual Counts for Bar Chart)
    counts = {
        "brain": scans.filter(scan_type='brain').count(),
        "bone": scans.filter(scan_type='bone').count(),
        "chest": scans.filter(scan_type='chest').count()
    }
    
    return render(request, "hospital_dashboard.html", {
        "stats": {
            "total": total_count,
            "today_scans": scans_today.count(),
            "critical_count": critical_scans.count(),
            "pending": pending_reviews,
            "approved": approved_count,
            "revenue": total_rev,
            "today_revenue": today_rev,
            "avg_accuracy": round(avg_confidence, 2),
            "doctors_online": User.objects.filter(profile__role='Doctor').count() + 1
        },
        "counts": counts,
        "critical_alerts": critical_scans[:3],
        "recent_scans": scans.order_by('-created_at')[:5]
    })

from django.contrib.auth.models import User
from .models import DoctorSettings, PrescriptionTemplate

def doctor_settings(request):
    if request.session.get('role') != 'admin':
        return redirect('admin_login')
    
    # Robust Demo Logic: Get or Create the admin user to avoid "DoesNotExist" error
    doctor_username = 'reddy'
    doctor, _ = User.objects.get_or_create(username=doctor_username, defaults={'is_staff': True})
    
    settings, created = DoctorSettings.objects.get_or_create(doctor=doctor)
    
    if request.method == "POST":
        # 1. Profile & Profile
        settings.specialization = request.POST.get('specialization', settings.specialization)
        settings.qualification = request.POST.get('qualification', settings.qualification)
        settings.registration_number = request.POST.get('registration_number', settings.registration_number)
        
        # 2. Availability
        settings.is_online = 'is_online' in request.POST
        settings.emergency_active = 'emergency_active' in request.POST
        settings.vacation_mode = 'vacation_mode' in request.POST
        settings.consultation_hours = request.POST.get('consultation_hours', settings.consultation_hours)
        
        # 3. AI Rules
        settings.confidence_threshold = int(request.POST.get('confidence_threshold', settings.confidence_threshold))
        settings.auto_approve_low_risk = 'auto_approve' in request.POST
        settings.show_heatmap = 'show_heatmap' in request.POST
        settings.qr_verification = 'qr_verification' in request.POST
        
        # 4. Patient Management
        settings.follow_up_days = int(request.POST.get('follow_up_days', settings.follow_up_days))
        settings.auto_archive = 'auto_archive' in request.POST
        settings.allow_reupload = 'allow_reupload' in request.POST
        settings.share_email = 'share_email' in request.POST
        
        # 5. Security
        settings.two_factor_auth = '2fa' in request.POST
        settings.session_timeout = int(request.POST.get('session_timeout', settings.session_timeout))
        settings.data_encryption = 'encryption' in request.POST
        
        # 6. Appearance
        settings.dark_mode = 'dark_mode' in request.POST
        settings.accent_color = request.POST.get('accent_color', settings.accent_color)
        settings.font_size = request.POST.get('font_size', settings.font_size)
        settings.compact_view = 'compact_view' in request.POST
        
        # 7. Billing
        settings.consultation_charge = request.POST.get('consultation_charge', settings.consultation_charge)
        settings.payout_method = request.POST.get('payout_method', settings.payout_method)
        
        # Files
        if 'signature' in request.FILES:
            settings.digital_signature = request.FILES['signature']
        if 'seal' in request.FILES:
            settings.hospital_seal = request.FILES['seal']
            
        settings.save()
        
        # Handle Template Creation
        if request.POST.get('template_title'):
            PrescriptionTemplate.objects.create(
                doctor=doctor,
                title=request.POST.get('template_title'),
                content=request.POST.get('template_content')
            )
            
        return redirect('doctor_settings')

    templates = PrescriptionTemplate.objects.filter(doctor=doctor)
    return render(request, "doctor_settings.html", {
        "settings": settings,
        "templates": templates
    })

def approve_report(request, id):
    scan = get_object_or_404(Scan, id=id)
    scan.status = 'APPROVED'
    scan.approval_timestamp = timezone.now()
    scan.save()
    return JsonResponse({"status": "success"})

def reject_report(request, id):
    scan = get_object_or_404(Scan, id=id)
    scan.status = 'REJECTED'
    scan.save()
    return JsonResponse({"status": "success"})

def delete_report(request, id):
    scan = get_object_or_404(Scan, id=id)
    # Optional: Delete associated media files here if necessary
    scan.delete()
    return JsonResponse({"status": "success"})

def upload(request):
    if request.session.get('role') != 'user':
        return redirect('user_login')
        
    if request.method == "POST" and request.FILES.get('image'):
        myfile = request.FILES['image']
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = os.path.join(settings.MEDIA_URL, 'uploads', filename)
        
        full_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
        scan_type = request.POST.get('scan_type')
        
        result = predict_image(full_path, filename=filename, media_root=settings.MEDIA_ROOT, scan_type=scan_type)
        
        if result.get("error"):
            return render(request, "dashboard.html", {
                "error_message": result["message"],
                "image_url": uploaded_file_url
            })
            
        pdf_filename = f"report_{os.path.splitext(filename)[0]}.pdf"
        pdf_relative_path = os.path.join('results', pdf_filename)
        pdf_full_path = os.path.join(settings.MEDIA_ROOT, pdf_relative_path)
        os.makedirs(os.path.dirname(pdf_full_path), exist_ok=True)
        
        patient_name = request.session.get('full_name', f"Patient_{Scan.objects.count() + 1}").title()
        if not patient_name.strip():
            patient_name = request.session.get('username', '').title()
            
        patient_id = f"AID-{1000 + Scan.objects.count() + 1}"
        
        generate_pdf(result, pdf_full_path, patient_name=patient_name, patient_id=patient_id)
        pdf_url = settings.MEDIA_URL + pdf_relative_path

        # GET SETTINGS FOR PERSONALIZED WORKFLOW
        from django.contrib.auth.models import User
        # The admin/doctor for settings
        doctor_user, _ = User.objects.get_or_create(username='reddy', defaults={'is_staff': True})
        doc_settings, _ = DoctorSettings.objects.get_or_create(doctor=doctor_user)
        
        # Determine the actual logged-in patient
        patient_username = request.session.get('username', 'demo_patient')
        patient_user, _ = User.objects.get_or_create(username=patient_username)
        
        # Determine Criticality based on Doctor's custom sensitivity
        ai_confidence = float(result['confidence'])
        is_emergency = result['requires_review'] or (ai_confidence < doc_settings.confidence_threshold)
        
        scan = Scan.objects.create(
            user=patient_user,
            image=os.path.join('uploads', filename),
            scan_type=scan_type,
            result_label=result['prediction'],
            confidence=ai_confidence,
            heatmap_url=result['heatmap_url'],
            boxed_url=result['boxed_url'],
            measurements=result['measurements'],
            clinical_note=result['clinical_note'],
            is_critical=is_emergency,
            status='AI_COMPLETED'
        )

        return render(request, "dashboard.html", {
            "result": result,
            "image_url": uploaded_file_url,
            "pdf_url": pdf_url,
            "scan_id": scan.id
        })

    return redirect('user_dashboard')

def patient_report_login(request):
    if request.method == "POST":
        username = request.POST.get('username', '').strip().lower()
        # In a real app, verify password here. For demo, we just accept the username to load their reports.
        request.session['report_username'] = username
        return redirect('patient_report_download')
    return render(request, "login.html", {"role": "report_download"})

def patient_report_download(request):
    username = request.session.get('report_username')
    if not username:
        return redirect('patient_report_login')
    
    # Get user object (creates if it doesn't exist for demo purposes)
    user, _ = User.objects.get_or_create(username=username)
    
    # Fetch ONLY approved scans
    approved_scans = Scan.objects.filter(user=user, status='APPROVED').order_by('-created_at')
    
    # Dynamically resolve PDF URLs since we save them to 'results/'
    for scan in approved_scans:
        base_name = os.path.splitext(os.path.basename(scan.image.name))[0]
        scan.pdf_url = f"/media/results/report_{base_name}.pdf"
        
    return render(request, "patient_report_download.html", {
        "scans": approved_scans,
        "username": username.capitalize()
    })

