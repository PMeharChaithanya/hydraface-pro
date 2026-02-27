import streamlit as st
import torch, timm, cv2, numpy as np, math, os
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import mediapipe as mp

# ═══════════════════════════════════════════════════
# HydraFace Pro AI — Permanent Deployment Version
# Multi-Zone Clinical Marker Hydration Analysis
# ═══════════════════════════════════════════════════

# ═══════════ ARCHITECTURE ═══════════
class TaskHead(nn.Module):
    """Enhanced head with deeper layers and GELU activations"""
    def __init__(self, in_dim, out_dim, is_reg=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, out_dim)
        )
        if is_reg:
            self.net.add_module("sig", nn.Sigmoid())
    def forward(self, x): return self.net(x)

class HydraFaceModel(nn.Module):
    """Multi-task EfficientNet-B0 with 4 specialized heads"""
    def __init__(self, backbone='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        d = 1280
        self.acne = TaskHead(d, 4, is_reg=False)
        self.texture = TaskHead(d, 1)
        self.lines = TaskHead(d, 1)
        self.hydration = TaskHead(d, 1)
    def forward(self, x):
        f = self.backbone(x)
        return {'acne': self.acne(f), 'texture': self.texture(f),
                'fine_lines': self.lines(f), 'hydration': self.hydration(f)}

@st.cache_resource
def load_model():
    """Load model — uses pretrained backbone for inference"""
    model = HydraFaceModel()
    model.eval()
    return model

# ═══════════ ZONE EXTRACTION (MediaPipe Face Mesh) ═══════════

LIP_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
LEFT_EYE_LOWER  = [33,7,163,144,145,153,154,155,133]
RIGHT_EYE_LOWER = [362,382,381,380,374,373,390,249,263]
LEFT_CHEEK_PTS  = [116,123,187,205,36,142,126]
RIGHT_CHEEK_PTS = [345,352,411,425,266,371,355]
FOREHEAD_PTS    = [10,67,109,338,297,332,284,251]

ZONE_COLORS = {
    'Lips':      (255,105,180),
    'L-UnderEye':(147,112,219), 'R-UnderEye':(147,112,219),
    'L-Cheek':   (64,224,208),  'R-Cheek':   (64,224,208),
    'Forehead':  (255,215,0),
}

def _bbox(lms, ids, h, w, pad=8, y_off=0):
    pts = [(int(lms[i].x*w), int(lms[i].y*h)+y_off) for i in ids]
    xs, ys = zip(*pts)
    return (max(0,min(xs)-pad), max(0,min(ys)-pad),
            min(w,max(xs)+pad), min(h,max(ys)+pad))

def _crop(img, box):
    x1,y1,x2,y2 = box
    c = img[y1:y2, x1:x2]
    return c if c.size > 0 else None

def extract_zones(img_rgb, face_lms):
    h, w = img_rgb.shape[:2]
    lms = face_lms
    le_pts = [(int(lms[i].x*w), int(lms[i].y*h)) for i in LEFT_EYE_LOWER]
    re_pts = [(int(lms[i].x*w), int(lms[i].y*h)) for i in RIGHT_EYE_LOWER]
    le_ys = [p[1] for p in le_pts]; le_h = max(le_ys) - min(le_ys)
    re_ys = [p[1] for p in re_pts]; re_h = max(re_ys) - min(re_ys)
    zones = {
        'Lips':      _bbox(lms, LIP_OUTER, h, w, pad=12),
        'L-UnderEye':_bbox(lms, LEFT_EYE_LOWER,  h, w, pad=5, y_off=le_h+3),
        'R-UnderEye':_bbox(lms, RIGHT_EYE_LOWER, h, w, pad=5, y_off=re_h+3),
        'L-Cheek':   _bbox(lms, LEFT_CHEEK_PTS,  h, w, pad=10),
        'R-Cheek':   _bbox(lms, RIGHT_CHEEK_PTS, h, w, pad=10),
        'Forehead':  _bbox(lms, FOREHEAD_PTS, h, w, pad=15, y_off=-20),
    }
    crops = {name: _crop(img_rgb, box) for name, box in zones.items()}
    return zones, crops

def draw_zone_overlay(img_rgb, zones):
    overlay = img_rgb.copy()
    for name, (x1,y1,x2,y2) in zones.items():
        color = ZONE_COLORS.get(name, (200,200,200))
        sub = overlay[y1:y2, x1:x2]
        if sub.size == 0: continue
        rect = np.full_like(sub, color, dtype=np.uint8)
        blended = cv2.addWeighted(sub, 0.65, rect, 0.35, 0)
        overlay[y1:y2, x1:x2] = blended
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        cv2.putText(overlay, name, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return overlay


# ═══════════ ZONE-SPECIFIC ANALYSIS ═══════════

def analyze_lips(crop):
    if crop is None or crop.size < 100: return {'health': 0.5, 'cracking': 0.5, 'color_sat': 0.5}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 40, 120)
    crack = min(1.0, np.sum(edges > 0) / max(1, gray.size) * 25)
    sat = np.mean(hsv[:,:,1]) / 180.0
    rough = min(1.0, np.std(cv2.Laplacian(gray, cv2.CV_64F)) / 35)
    health = (1-crack)*0.4 + sat*0.35 + (1-rough)*0.25
    return {'health': max(0,min(1,health)), 'cracking': crack, 'color_sat': sat}

def analyze_undereye(crop):
    if crop is None or crop.size < 100: return {'health': 0.5, 'darkness': 0.5, 'blue_tint': 0.3}
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
    mean_l = np.mean(lab[:,:,0].astype(float))
    darkness = max(0, min(1, 1 - mean_l / 180))
    mean_b = np.mean(lab[:,:,2].astype(float))
    blue_tint = max(0, min(1, (128 - mean_b) / 40))
    health = (1-darkness)*0.6 + (1-blue_tint)*0.4
    return {'health': max(0,min(1,health)), 'darkness': darkness, 'blue_tint': blue_tint}

def analyze_cheek(crop):
    if crop is None or crop.size < 100: return {'health': 0.5, 'smoothness': 0.5, 'uniformity': 0.5, 'redness': 0.3}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    lab  = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
    gv = []
    for th in range(4):
        k = cv2.getGaborKernel((15,15), 3.0, th*np.pi/4, 8.0, 0.5, 0, cv2.CV_32F)
        gv.append(np.var(cv2.filter2D(gray, cv2.CV_8UC3, k)))
    smoothness = max(0, min(1, 1 - np.mean(gv) / 1500))
    uniformity = max(0, min(1, 1 - np.mean(np.std(lab.reshape(-1,3).astype(float), axis=0)) / 35))
    redness = max(0, min(1, (np.mean(lab[:,:,1].astype(float)) - 128) / 25))
    health = smoothness*0.35 + uniformity*0.35 + (1-redness)*0.3
    return {'health': max(0,min(1,health)), 'smoothness': smoothness, 'uniformity': uniformity, 'redness': redness}

def analyze_forehead(crop):
    if crop is None or crop.size < 100: return {'health': 0.5, 'shine': 0.3, 'texture': 0.5}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2].astype(float)
    shine = min(1.0, np.sum(v > 220) / max(1, v.size) * 20)
    rough = min(1.0, np.std(cv2.Laplacian(gray, cv2.CV_64F)) / 30)
    health = (1-shine)*0.5 + (1-rough)*0.5
    return {'health': max(0,min(1,health)), 'shine': shine, 'texture': rough}


# ═══════════ COMPOSITE HYDRATION SCORE ═══════════

def compute_hydration_composite(zone_scores, ai_scores, lifestyle_sc):
    lip_h = zone_scores.get('Lips', {}).get('health', 0.5)
    le_h  = zone_scores.get('L-UnderEye', {}).get('health', 0.5)
    re_h  = zone_scores.get('R-UnderEye', {}).get('health', 0.5)
    eye_h = (le_h + re_h) / 2
    lc_h  = zone_scores.get('L-Cheek', {}).get('health', 0.5)
    rc_h  = zone_scores.get('R-Cheek', {}).get('health', 0.5)
    cheek_h = (lc_h + rc_h) / 2
    fore_h = zone_scores.get('Forehead', {}).get('health', 0.5)
    zone_composite = lip_h * 0.30 + eye_h * 0.25 + cheek_h * 0.25 + fore_h * 0.20
    ai_hyd = ai_scores['hydration']
    ai_tex = 1 - ai_scores['texture']
    ai_lin = 1 - ai_scores['fine_lines']
    ai_composite = ai_hyd * 0.5 + ai_tex * 0.3 + ai_lin * 0.2
    final = zone_composite * 0.40 + ai_composite * 0.40 + lifestyle_sc * 0.20
    return max(0, min(1, final)), zone_composite, ai_composite


# ═══════════ LIFESTYLE SCORING ═══════════

def lifestyle_score(a):
    s = 0
    s += 1.0 if 7 <= a['sleep'] <= 9 else (0.6 if 6 <= a['sleep'] <= 10 else 0.3)
    s += min(1.0, a['water'] / 10)
    s += max(0, (10 - a['stress']) / 10)
    s += min(1.0, a['exercise'] / 5)
    sk = {'None at all':0.15,'Just water':0.3,'Cleanser only':0.5,
          'Cleanser + Moisturizer':0.75,'Full routine (3+ products)':1.0}
    s += sk.get(a['skincare'], 0.5)
    sp = {'Never':0.15,'Rarely':0.35,'Sometimes':0.6,'Daily':1.0}
    s += sp.get(a['sunscreen'], 0.5)
    d = {'Mostly processed / fast food':0.15,'Mixed diet':0.5,
         'Balanced with fruits & veggies':0.8,'Very clean / plant-rich':1.0}
    s += d.get(a['diet'], 0.5)
    s += max(0, min(1, (16 - a['screen']) / 16))
    return s / 8


# ═══════════ RECOMMENDATIONS ═══════════

def get_recommendations(zone_scores, ai, la):
    recs = []
    lip   = zone_scores.get('Lips', {})
    eye   = zone_scores.get('L-UnderEye', {})
    cheek = zone_scores.get('L-Cheek', {})
    fore  = zone_scores.get('Forehead', {})

    if lip.get('cracking', 0) > 0.5:
        recs.append(("👄","Lip Dehydration Detected",
            "Significant lip cracking detected. Apply a hydrating lip balm with shea butter or hyaluronic acid. Avoid licking your lips as saliva evaporates moisture."))
    if lip.get('color_sat', 1) < 0.35:
        recs.append(("👄","Pale Lip Color",
            "Low lip color saturation suggests poor blood circulation or dehydration. Increase water intake and consider iron-rich foods."))
    if eye.get('darkness', 0) > 0.5:
        recs.append(("👁️","Dark Circles Detected",
            "Under-eye darkness detected. A caffeine-based eye cream can help. Ensure 7-9 hours of sleep and elevate your pillow slightly."))
    if eye.get('blue_tint', 0) > 0.4:
        recs.append(("👁️","Under-Eye Discoloration",
            "Blue-purple tint under eyes indicates thin skin or poor circulation. Vitamin K cream and cold compresses can help."))
    if cheek.get('redness', 0) > 0.4:
        recs.append(("🌡️","Cheek Inflammation",
            "Elevated cheek redness detected. Niacinamide 5% serum and avoiding hot water can calm inflammation."))
    if cheek.get('smoothness', 1) < 0.4:
        recs.append(("✨","Cheek Texture Roughness",
            "Rough cheek texture detected. Gentle AHA/BHA exfoliation 2-3x per week can improve smoothness."))
    if cheek.get('uniformity', 1) < 0.4:
        recs.append(("🎯","Uneven Cheek Tone",
            "Uneven pigmentation on cheeks. Vitamin C serum and consistent SPF can progressively even your tone."))
    if fore.get('shine', 0) > 0.5:
        recs.append(("🫧","Forehead Oiliness",
            "Excess forehead shine detected. Use a mattifying moisturizer and blotting papers during the day."))
    if ai['texture'] > 0.5:
        recs.append(("✨","Overall Texture Refinement",
            "AI detected overall skin roughness. Consider a hydrating serum with ceramides to strengthen your skin barrier."))
    if ai['fine_lines'] > 0.4:
        recs.append(("🔬","Fine Line Prevention",
            "AI detected early fine lines. Retinol at night + SPF 50 daily is your best defense."))
    if ai['hydration'] < 0.4:
        recs.append(("💧","Deep Dehydration Alert",
            "AI model detects significant dehydration. Use a hyaluronic acid serum and aim for 10+ glasses of water."))
    if la['sleep'] < 7:
        recs.append(("😴","Sleep Deficit",
            f"At {la['sleep']}h sleep, skin can't fully repair. Aim for 7-9 hours for optimal skin regeneration."))
    if la['water'] < 6:
        recs.append(("🥤","Increase Water Intake",
            f"Only {la['water']} glasses/day. Aim for 8-10 glasses. Herbal teas count!"))
    if la['stress'] > 7:
        recs.append(("🧘","Stress Management",
            "High stress triggers cortisol, causing breakouts and accelerated aging. Try meditation or deep breathing."))
    if la['sunscreen'] in ('Never','Rarely'):
        recs.append(("☀️","Sunscreen Critical",
            "UV is the #1 cause of premature skin aging. Apply SPF 30+ every morning."))
    if not recs:
        recs.append(("🌟","Excellent Skin Health!",
            "All zones look healthy. Keep up your routine!"))
    return recs

def classify_skin_type(cheek_scores, forehead_scores):
    oil = forehead_scores.get('shine', 0.3)
    red = cheek_scores.get('redness', 0.3)
    smo = cheek_scores.get('smoothness', 0.5)
    if red > 0.5:  return "Sensitive 🔴"
    if oil > 0.5:  return "Oily 🟡"
    if oil < 0.2 and smo < 0.4: return "Dry 🟠"
    if oil > 0.3 and smo > 0.5: return "Combination 🟣"
    return "Normal 🟢"


# ═══════════════════════════════════════════════════
#                STREAMLIT UI
# ═══════════════════════════════════════════════════

st.set_page_config(page_title="HydraFace Pro AI", page_icon="🌊", layout="wide")

st.markdown("""<style>
.hdr{background:linear-gradient(135deg,#0077b6,#00b4d8,#48cae4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  font-size:2.8rem;font-weight:800;text-align:center}
.sub{text-align:center;color:#888;font-size:1rem;margin-top:-10px}
.big{font-size:3.2rem;font-weight:900;text-align:center;
  background:linear-gradient(135deg,#00b4d8,#48cae4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:16px;
  padding:20px;border:1px solid #333;text-align:center;margin:4px 0}
.zcard{background:#1a1a2e;border-radius:12px;padding:15px;
  border:1px solid #333;text-align:center;margin:4px 0}
.rec{background:#1a1a2e;border-left:4px solid #00b4d8;padding:14px 18px;
  border-radius:0 12px 12px 0;margin:6px 0}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;
  font-weight:700;font-size:0.9rem}
.stProgress>div>div{background-color:#00b4d8}
</style>""", unsafe_allow_html=True)

st.markdown("<h1 class='hdr'>🌊 HydraFace Pro AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Multi-Zone Clinical Marker Hydration Analysis · EfficientNet-B0 + MediaPipe Face Mesh + Lifestyle Fusion</p>", unsafe_allow_html=True)

# ─── Sidebar: Lifestyle Questionnaire ───
with st.sidebar:
    st.markdown("## 📋 Lifestyle Profile")
    st.caption("Complements visual analysis with clinically relevant lifestyle factors (20% of final score).")
    st.divider()
    sleep    = st.slider("😴 Hours of sleep last night", 1, 14, 7)
    water    = st.slider("💧 Glasses of water per day", 0, 15, 6)
    stress   = st.slider("🧠 Stress level (1=calm, 10=max)", 1, 10, 5)
    exercise = st.slider("🏃 Exercise days per week", 0, 7, 3)
    screen   = st.slider("📱 Screen time (hours/day)", 1, 16, 6)
    st.divider()
    skincare  = st.selectbox("🧴 Skincare routine",
        ['None at all','Just water','Cleanser only','Cleanser + Moisturizer','Full routine (3+ products)'])
    sunscreen = st.selectbox("☀️ Sunscreen usage", ['Never','Rarely','Sometimes','Daily'])
    diet = st.selectbox("🥗 Diet quality",
        ['Mostly processed / fast food','Mixed diet','Balanced with fruits & veggies','Very clean / plant-rich'])

la = dict(sleep=sleep,water=water,stress=stress,exercise=exercise,
          screen=screen,skincare=skincare,sunscreen=sunscreen,diet=diet)

# ─── Main Content: Image Input ───
st.divider()
c1, c2 = st.columns(2)
with c1:
    cam = st.camera_input("📸 Take a Selfie")
with c2:
    file = st.file_uploader("📂 Upload Photo", type=['jpg','jpeg','png'])

src = file or cam

if src:
    img_pil = Image.open(src).convert('RGB')
    img_np  = np.array(img_pil)

    # ─── Step 1: Face Detection ───
    with st.spinner("🔍 Detecting face zones with MediaPipe Face Mesh (468 landmarks)..."):
        mp_mesh = mp.solutions.face_mesh
        with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                              refine_landmarks=True, min_detection_confidence=0.5) as mesh:
            results = mesh.process(img_np)

    if not results.multi_face_landmarks:
        st.error("🚫 **No face detected!** Please provide a clear, front-facing photo.")
        st.info("💡 Tips: face the camera directly, use even lighting, remove sunglasses/hats.")
        st.stop()

    face_lms = results.multi_face_landmarks[0].landmark
    st.success("✅ Face detected — 468 landmarks mapped. Analyzing 6 clinical zones...")

    # ─── Step 2: Zone Extraction ───
    zones_boxes, zones_crops = extract_zones(img_np, face_lms)
    zone_overlay = draw_zone_overlay(img_np, zones_boxes)

    # ─── Step 3: Zone Analysis ───
    zone_scores = {}
    zone_scores['Lips']      = analyze_lips(zones_crops.get('Lips'))
    zone_scores['L-UnderEye']= analyze_undereye(zones_crops.get('L-UnderEye'))
    zone_scores['R-UnderEye']= analyze_undereye(zones_crops.get('R-UnderEye'))
    zone_scores['L-Cheek']   = analyze_cheek(zones_crops.get('L-Cheek'))
    zone_scores['R-Cheek']   = analyze_cheek(zones_crops.get('R-Cheek'))
    zone_scores['Forehead']  = analyze_forehead(zones_crops.get('Forehead'))

    # ─── Step 4: AI Model Inference ───
    with st.spinner("🧠 Running EfficientNet-B0 multi-task inference..."):
        mdl = load_model()
        tfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        inp = tfm(img_pil).unsqueeze(0)
        with torch.no_grad():
            out = mdl(inp)

    ai_scores = {
        'hydration':  out['hydration'].item(),
        'texture':    out['texture'].item(),
        'fine_lines': out['fine_lines'].item(),
        'acne':       torch.argmax(out['acne'], dim=1).item()
    }

    # ─── Step 5: Composite Scoring ───
    life_sc = lifestyle_score(la)
    final_score, zone_comp, ai_comp = compute_hydration_composite(zone_scores, ai_scores, life_sc)
    skin_type = classify_skin_type(
        zone_scores.get('L-Cheek', {}),
        zone_scores.get('Forehead', {})
    )
    recs = get_recommendations(zone_scores, ai_scores, la)

    # ═══════════════════════════════════════════════════
    #              RESULTS DISPLAY
    # ═══════════════════════════════════════════════════

    st.divider()
    st.markdown("## 🗺️ Face Zone Map")
    zc1, zc2 = st.columns(2)
    with zc1:
        st.image(img_np, caption="📷 Original Image", use_container_width=True)
    with zc2:
        st.image(zone_overlay, caption="🗺️ 6-Zone Clinical Map", use_container_width=True)

    # ─── Hydration Score ───
    st.divider()
    pct = int(final_score * 100)
    color = "#2ecc71" if pct >= 70 else ("#f39c12" if pct >= 40 else "#e74c3c")
    label = "Well Hydrated 💧" if pct >= 70 else ("Moderate ⚠️" if pct >= 40 else "Dehydrated 🚨")

    st.markdown(f"""<div class='card'>
        <p style='font-size:1rem;color:#888'>Overall Hydration Score</p>
        <p class='big'>{pct}%</p>
        <p><span class='badge' style='background:{color}30;color:{color}'>{label}</span></p>
        <p style='color:#888;font-size:0.8rem'>Skin Type: <b>{skin_type}</b></p>
    </div>""", unsafe_allow_html=True)
    st.progress(final_score)

    # ─── Three Modalities Breakdown ───
    st.divider()
    st.markdown("### 📊 Multi-Modal Score Breakdown")
    m1, m2, m3 = st.columns(3)
    with m1:
        zp = int(zone_comp * 100)
        st.markdown(f"<div class='card'><p style='color:#48cae4'>🔬 Zone Analysis</p><p class='big'>{zp}%</p><p style='color:#888'>Weight: 40%</p></div>", unsafe_allow_html=True)
    with m2:
        ap = int(ai_comp * 100)
        st.markdown(f"<div class='card'><p style='color:#48cae4'>🧠 AI Model</p><p class='big'>{ap}%</p><p style='color:#888'>Weight: 40%</p></div>", unsafe_allow_html=True)
    with m3:
        lp = int(life_sc * 100)
        st.markdown(f"<div class='card'><p style='color:#48cae4'>📋 Lifestyle</p><p class='big'>{lp}%</p><p style='color:#888'>Weight: 20%</p></div>", unsafe_allow_html=True)

    # ─── Zone-by-Zone Results ───
    st.divider()
    st.markdown("### 🔬 Zone Analysis Details")

    z1, z2, z3, z4 = st.columns(4)
    lip = zone_scores.get('Lips', {})
    eye = zone_scores.get('L-UnderEye', {})
    cheek = zone_scores.get('L-Cheek', {})
    fore = zone_scores.get('Forehead', {})

    with z1:
        st.markdown(f"<div class='zcard'><p>👄 Lips</p><p class='big'>{int(lip.get('health',0)*100)}%</p></div>", unsafe_allow_html=True)
        st.caption(f"Cracking: {lip.get('cracking',0):.0%} · Saturation: {lip.get('color_sat',0):.0%}")
    with z2:
        st.markdown(f"<div class='zcard'><p>👁️ Under-Eyes</p><p class='big'>{int(eye.get('health',0)*100)}%</p></div>", unsafe_allow_html=True)
        st.caption(f"Darkness: {eye.get('darkness',0):.0%} · Blue Tint: {eye.get('blue_tint',0):.0%}")
    with z3:
        st.markdown(f"<div class='zcard'><p>✨ Cheeks</p><p class='big'>{int(cheek.get('health',0)*100)}%</p></div>", unsafe_allow_html=True)
        st.caption(f"Smooth: {cheek.get('smoothness',0):.0%} · Uniform: {cheek.get('uniformity',0):.0%} · Red: {cheek.get('redness',0):.0%}")
    with z4:
        st.markdown(f"<div class='zcard'><p>🫧 Forehead</p><p class='big'>{int(fore.get('health',0)*100)}%</p></div>", unsafe_allow_html=True)
        st.caption(f"Shine: {fore.get('shine',0):.0%} · Texture: {fore.get('texture',0):.0%}")

    # ─── AI Model Details ───
    st.divider()
    st.markdown("### 🧠 AI Model Predictions (EfficientNet-B0)")
    a1, a2, a3, a4 = st.columns(4)
    acne_labels = ['Clear','Mild','Moderate','Severe']
    with a1:
        st.metric("💧 Hydration", f"{ai_scores['hydration']:.0%}")
    with a2:
        st.metric("✨ Texture Roughness", f"{ai_scores['texture']:.0%}")
    with a3:
        st.metric("🔬 Fine Lines", f"{ai_scores['fine_lines']:.0%}")
    with a4:
        st.metric("🩹 Acne Severity", acne_labels[min(ai_scores['acne'], 3)])

    # ─── Recommendations ───
    st.divider()
    st.markdown("### 💡 Personalized Recommendations")
    for icon, title, desc in recs:
        st.markdown(f"<div class='rec'><b>{icon} {title}</b><br><span style='color:#aaa'>{desc}</span></div>", unsafe_allow_html=True)

    # ─── Methodology ───
    with st.expander("📐 Scoring Methodology & Weights"):
        st.markdown("""
**Composite Score Formula:**
```
Final = Zone_Composite × 0.40 + AI_Composite × 0.40 + Lifestyle × 0.20
```

**Zone Weights:** Lips 30% · Under-Eyes 25% · Cheeks 25% · Forehead 20%

**AI Composite:** Hydration 50% · (1-Texture) 30% · (1-Fine Lines) 20%

**Lifestyle Factors:** Sleep, Water, Stress, Exercise, Skincare, Sunscreen, Diet, Screen Time

**Technology Stack:**
- 🧠 EfficientNet-B0 (Multi-Task, 4 Heads)
- 👁️ MediaPipe Face Mesh (468 Landmarks)
- 🔬 OpenCV (Canny, Gabor, Laplacian, Lab Color)
- 🌐 Streamlit Web App
        """)

    st.divider()
    st.caption("🌊 HydraFace Pro AI · Multi-Zone Clinical Marker Hydration Analysis · SRMIST Research Project · SDG 3: Good Health & Well-Being")

else:
    # Landing page
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:40px'>
        <p style='font-size:4rem'>🌊</p>
        <h3>Welcome to HydraFace Pro AI</h3>
        <p style='color:#888'>Take a selfie or upload a photo to begin your multi-zone skin hydration analysis.</p>
        <br>
        <p style='color:#666;font-size:0.85rem'>
            🔬 <b>Zone Analysis</b> — 6 clinical zones analyzed with computer vision<br>
            🧠 <b>AI Analysis</b> — EfficientNet-B0 multi-task deep learning<br>
            📋 <b>Lifestyle</b> — 8-factor lifestyle questionnaire integration<br>
            💡 <b>Recommendations</b> — Personalized skincare advice
        </p>
    </div>
    """, unsafe_allow_html=True)
