from django.db import models
from django.contrib.auth.models import User

# Startup Level: Hospital Management
class Hospital(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    license_key = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

# Startup Level: Multi-Role System
class UserProfile(models.Model):
    ROLES = [
        ('Patient', 'Patient'),
        ('Doctor', 'Doctor'),
        ('Admin', 'Hospital Admin'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLES, default='Patient')
    hospital = models.ForeignKey(Hospital, on_delete=models.SET_NULL, null=True, blank=True)
    phone = models.CharField(max_length=15, blank=True)
    
    def __str__(self):
        return f"{self.user.username} ({self.role})"

# Startup Level: Smart Workflow Scan Model
class Scan(models.Model):
    STATUS_CHOICES = [
        ('AI_COMPLETED', 'AI Assessment Done'),
        ('PENDING_REVIEW', 'Doctor Review Pending'),
        ('APPROVED', 'Verified by Doctor'),
        ('CRITICAL', 'Emergency Alert!'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scans')
    hospital = models.ForeignKey(Hospital, on_delete=models.SET_NULL, null=True, blank=True)
    assigned_doctor = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='assigned_reviews')
    
    image = models.ImageField(upload_to='scans/')
    scan_type = models.CharField(max_length=20) # brain, bone, chest
    
    # AI Results
    result_label = models.CharField(max_length=100, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    heatmap_url = models.CharField(max_length=255, null=True, blank=True)
    boxed_url = models.CharField(max_length=255, null=True, blank=True)
    
    # Measurements (JSON for flexibility)
    measurements = models.JSONField(default=dict, blank=True)
    clinical_note = models.TextField(blank=True)
    
    # Workflow
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='AI_COMPLETED')
    is_critical = models.BooleanField(default=False)
    
    # Doctor Validation
    doctor_signature = models.ImageField(upload_to='signatures/', null=True, blank=True)
    approval_timestamp = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Scan for {self.user.username} - {self.status}"

# Startup Level: Advanced Doctor Workflow Settings
class DoctorSettings(models.Model):
    doctor = models.OneToOneField(User, on_delete=models.CASCADE, related_name='settings')
    
    # 1. Profile & Professional
    specialization = models.CharField(max_length=100, blank=True)
    qualification = models.CharField(max_length=100, blank=True)
    registration_number = models.CharField(max_length=50, blank=True)
    digital_signature = models.ImageField(upload_to='signatures/', null=True, blank=True)
    hospital_seal = models.ImageField(upload_to='seals/', null=True, blank=True)
    
    # 2. Availability Settings
    is_online = models.BooleanField(default=True)
    emergency_active = models.BooleanField(default=True)
    consultation_hours = models.CharField(max_length=100, default="09:00 AM - 05:00 PM")
    vacation_mode = models.BooleanField(default=False)
    
    # 3. AI & Review Settings
    confidence_threshold = models.IntegerField(default=85)
    auto_approve_low_risk = models.BooleanField(default=False)
    show_heatmap = models.BooleanField(default=True)
    qr_verification = models.BooleanField(default=True)
    
    # 4. Patient Management (NEW)
    follow_up_days = models.IntegerField(default=7)
    auto_archive = models.BooleanField(default=True)
    allow_reupload = models.BooleanField(default=True)
    share_email = models.BooleanField(default=True)
    
    # 5. Privacy & Security (NEW)
    two_factor_auth = models.BooleanField(default=False)
    session_timeout = models.IntegerField(default=30) # in minutes
    data_encryption = models.BooleanField(default=True)
    
    # 6. Appearance Settings (NEW)
    dark_mode = models.BooleanField(default=True)
    accent_color = models.CharField(max_length=20, default="#00f2ff")
    font_size = models.CharField(max_length=10, default="Normal") # Small, Normal, Large
    compact_view = models.BooleanField(default=False)
    
    # 7. Billing / Earnings (NEW)
    consultation_charge = models.DecimalField(max_digits=10, decimal_places=2, default=500.00)
    payout_method = models.CharField(max_length=50, default="Bank Transfer")
    
    def __str__(self):
        return f"Settings for Dr. {self.doctor.username}"

class PrescriptionTemplate(models.Model):
    doctor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='templates')
    title = models.CharField(max_length=100) # e.g., "Bone Fracture Advice"
    content = models.TextField()
    
    def __str__(self):
        return self.title
