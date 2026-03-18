from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
import json
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
import os
import requests
from django.views.decorators.csrf import csrf_exempt
import google.generativeai as genai
from PIL import Image
import torch
from torchvision import transforms
from django.conf import settings


def home(request):
    return render(request, 'dashboard/home.html')


val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(settings.BASE_DIR,'model','Plant_disease_model.pt')
soil_model_path = os.path.join(settings.BASE_DIR,'model','soil_classification.pt')
try:
    pytorch_model = torch.jit.load(model_path, map_location=device)
    pytorch_model.eval()
    print("✅ PyTorch model loaded successfully")
except Exception as e:
    pytorch_model = None
    print(f"❌ Failed to load PyTorch model: {e}")
    print("Disease detection will not work until model file is provided.")

try:
    soil_model = torch.jit.load(soil_model_path, map_location=device)
    soil_model.eval()
    print("✅ Soil classification model loaded successfully")
except Exception as e:
    soil_model = None
    print(f"❌ Failed to load soil classification model: {e}")
    print("Soil classification will not work until model file is provided.")

def predict_image(image_path, pytorch_model, class_names):
    image = Image.open(image_path).convert("RGB")
    image = val_test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = pytorch_model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

def predict_soil_type(image_path, soil_model):
    """Predict soil type from image"""
    if soil_model is None:
        return "Soil classification model is not available. Please contact administrator."
    
    soil_classes = [
        'Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil', 'Mountain_Soil', 'Red_Soil', 'Yellow_Soil'
    ]
    
    image = Image.open(image_path).convert("RGB")
    image = val_test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = soil_model(image)
        _, predicted = torch.max(outputs, 1)
    return soil_classes[predicted.item()]

def get_soil_info(soil_type):
    """Get detailed information about soil type"""
    soil_info = {
        'Alluvial_Soil': {
            'description': 'Fertile soil deposited by rivers, rich in minerals and nutrients.',
            'characteristics': [
                'Highly fertile and productive',
                'Good water retention capacity',
                'Suitable for wide range of crops',
                'Found in river valleys and flood plains',
                'Contains silt, clay and sand particles'
            ],
            'best_crops': ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize'],
            'management': [
                'Regular irrigation required',
                'Organic matter addition beneficial',
                'Proper drainage to avoid waterlogging',
                'Balanced fertilization',
                'Crop rotation recommended'
            ]
        },
        'Arid_Soil': {
            'description': 'Soil found in dry, arid regions with low moisture content and limited vegetation.',
            'characteristics': [
                'Low organic matter content',
                'Poor water retention',
                'High mineral content',
                'Subject to wind erosion',
                'Alkaline pH in many cases'
            ],
            'best_crops': ['Millets', 'Sorghum', 'Pearl millet', 'Groundnut', 'Sesame'],
            'management': [
                'Conserve soil moisture',
                'Use drought-resistant varieties',
                'Apply organic mulch',
                'Practice water harvesting',
                'Avoid deep ploughing'
            ]
        },
        'Black_Soil': {
            'description': 'Dark colored, clay-rich soil known for high fertility and moisture retention.',
            'characteristics': [
                'High water holding capacity',
                'Rich in calcium carbonate',
                'Expands when wet, cracks when dry',
                'Very fertile for agriculture',
                'Also known as Regur soil'
            ],
            'best_crops': ['Cotton', 'Sugarcane', 'Groundnut', 'Wheat', 'Soybean'],
            'management': [
                'Deep ploughing during summer',
                'Conserve moisture for dry periods',
                'Apply organic manure regularly',
                'Avoid waterlogging',
                'Use gypsum for reclamation if needed'
            ]
        },
        'Laterite_Soil': {
            'description': 'Soil formed in hot, humid tropical regions with high iron and aluminum content.',
            'characteristics': [
                'Reddish color due to iron oxides',
                'Low fertility',
                'Acidic to neutral pH',
                'Poor water retention',
                'Hardens when exposed to air'
            ],
            'best_crops': ['Tea', 'Coffee', 'Rubber', 'Cashews', 'Pineapple'],
            'management': [
                'Liming to correct acidity',
                'Regular addition of organic matter',
                'Conservative tillage practices',
                'Use of cover crops',
                'Balanced fertilization'
            ]
        },
        'Mountain_Soil': {
            'description': 'Soil found in mountainous regions, often shallow and rocky with good drainage.',
            'characteristics': [
                'Shallow depth',
                'Rocky and stony',
                'Good drainage',
                'Variable fertility',
                'Subject to erosion'
            ],
            'best_crops': ['Potatoes', 'Apples', 'Maize', 'Wheat', 'Barley'],
            'management': [
                'Contour farming to prevent erosion',
                'Use of organic matter',
                'Conservative tillage',
                'Terracing on slopes',
                'Crop rotation'
            ]
        },
        'Red_Soil': {
            'description': 'Reddish colored soil formed due to weathering of igneous rocks, acidic in nature.',
            'characteristics': [
                'Acidic pH (5.5-6.5)',
                'Rich in iron and aluminum oxides',
                'Low fertility compared to others',
                'Good drainage',
                'Easily eroded on slopes'
            ],
            'best_crops': ['Groundnut', 'Millets', 'Cotton', 'Sugarcane', 'Tobacco'],
            'management': [
                'Liming to correct acidity',
                'Add organic matter regularly',
                'Conserve soil moisture',
                'Contour farming on slopes',
                'Balanced fertilization with micronutrients'
            ]
        },
        'Yellow_Soil': {
            'description': 'Light colored soil found in humid subtropical regions with moderate fertility.',
            'characteristics': [
                'Light yellowish color',
                'Moderate fertility',
                'Good drainage',
                'Sandy loam texture',
                'Low clay content'
            ],
            'best_crops': ['Rice', 'Wheat', 'Sugarcane', 'Soybean', 'Vegetables'],
            'management': [
                'Regular irrigation',
                'Organic matter addition',
                'Balanced fertilization',
                'Crop rotation',
                'Conserve soil moisture'
            ]
        }
    }
    
    return soil_info.get(soil_type, {
        'description': 'Unknown soil type',
        'characteristics': ['Information not available'],
        'best_crops': ['Please consult local agricultural extension'],
        'management': ['Please consult local agricultural extension']
    })

# Comprehensive Plant Disease Solutions Dictionary
PLANT_DISEASE_SOLUTIONS = {
    'Apple___Apple_scab': {
        'disease_name': 'Apple Scab',
        'affected_crop': 'Apple',
        'severity': 'High',
        'symptoms': [
            'Dark, circular lesions on leaves',
            'Olive-brown spots on fruit',
            'Cracked, corky appearance on fruit',
            'Premature leaf drop',
            'Deformed fruit'
        ],
        'causes': 'Fungal infection (Venturia inaequalis)',
        'solutions': [
            'Apply fungicides (sulfur or captan) during growing season',
            'Remove and destroy infected leaves and fruit',
            'Improve air circulation by pruning',
            'Avoid overhead watering',
            'Plant resistant apple varieties',
            'Apply preventive sprays in spring',
            'Maintain proper spacing between trees'
        ],
        'prevention': 'Use disease-resistant varieties, maintain orchard hygiene, apply dormant oil in winter'
    },
    
    'Apple___Black_rot': {
        'disease_name': 'Black Rot',
        'affected_crop': 'Apple',
        'severity': 'High',
        'symptoms': [
            'Large, dark, sunken lesions on fruit',
            'Concentric rings on rotted fruit',
            'Fruit mummification',
            'Cankers on branches and trunk',
            'Leaf spots with red halos'
        ],
        'causes': 'Fungal infection (Botryosphaeria obtusa)',
        'solutions': [
            'Prune and remove infected branches',
            'Destroy mummified fruit and dead wood',
            'Apply copper or sulfur fungicides',
            'Improve tree vigor with proper nutrition',
            'Ensure good air circulation',
            'Remove cankers from tree trunk',
            'Sanitize pruning tools between cuts'
        ],
        'prevention': 'Remove dead wood, maintain tree health, apply dormant sprays, improve drainage'
    },
    
    'Apple___Cedar_apple_rust': {
        'disease_name': 'Cedar Apple Rust',
        'affected_crop': 'Apple',
        'severity': 'Medium',
        'symptoms': [
            'Orange-yellow spots on leaves and fruit',
            'Tube-like projections on fruit',
            'Yellow halos around spots',
            'Deformed fruit',
            'Premature defoliation'
        ],
        'causes': 'Fungal infection (Gymnosporangium juniperi-virginianae)',
        'solutions': [
            'Remove cedar/juniper trees within 1-2 miles if possible',
            'Apply fungicides (myclobutanil or mancozeb) every 10-14 days',
            'Start spraying when cedar trees produce galls',
            'Use resistant apple varieties',
            'Improve air circulation',
            'Remove infected leaves and fruit promptly',
            'Apply preventive sprays before infection period'
        ],
        'prevention': 'Plant resistant varieties, maintain distance from cedar/juniper hosts, apply preventive fungicides'
    },
    
    'Apple___healthy': {
        'disease_name': 'Healthy Apple',
        'affected_crop': 'Apple',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Continue regular monitoring',
            'Maintain proper irrigation (1-2 inches per week)',
            'Apply balanced fertilizer (10-10-10 NPK)',
            'Prune to maintain tree shape and air circulation',
            'Monitor for pests and diseases'
        ],
        'prevention': 'Regular inspection, proper cultural practices, adequate nutrition'
    },
    
    'Blueberry___healthy': {
        'disease_name': 'Healthy Blueberry',
        'affected_crop': 'Blueberry',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Maintain consistent moisture (1-2 inches per week)',
            'Keep soil pH between 4.5-5.5',
            'Apply acidifying fertilizer if needed',
            'Mulch with pine bark or wood chips',
            'Monitor for insects and diseases'
        ],
        'prevention': 'Proper soil pH, adequate drainage, mulching, regular monitoring'
    },
    
    'Cherry_(including_sour)___Powdery_mildew': {
        'disease_name': 'Powdery Mildew on Cherry',
        'affected_crop': 'Cherry (including sour)',
        'severity': 'Medium',
        'symptoms': [
            'White, powdery coating on leaves',
            'Distorted leaf growth',
            'Curled or deformed leaves',
            'Powdery coating on stems and fruit',
            'Reduced fruit quality'
        ],
        'causes': 'Fungal infection (Podosphaera clandestina)',
        'solutions': [
            'Apply sulfur fungicides (not in hot weather above 85°F)',
            'Use potassium bicarbonate sprays',
            'Remove infected leaves and branches',
            'Improve air circulation by pruning',
            'Avoid overhead watering',
            'Apply neem oil sprays',
            'Use resistant cherry varieties'
        ],
        'prevention': 'Plant resistant varieties, maintain air circulation, reduce humidity, avoid excess nitrogen'
    },
    
    'Cherry_(including_sour)___healthy': {
        'disease_name': 'Healthy Cherry',
        'affected_crop': 'Cherry (including sour)',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water consistently (1-2 inches per week)',
            'Apply balanced fertilizer in spring',
            'Prune to maintain shape and air circulation',
            'Monitor for pests and diseases regularly',
            'Thin fruit to improve size'
        ],
        'prevention': 'Proper care, monitoring, adequate nutrition'
    },
    
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'disease_name': 'Cercospora Leaf Spot / Gray Leaf Spot',
        'affected_crop': 'Corn (maize)',
        'severity': 'High',
        'symptoms': [
            'Rectangular, gray-brown lesions on leaves',
            'Lesions surrounded by yellow or red halos',
            'Lesions coalesce causing leaf death',
            'Starts on lower leaves and progresses upward',
            'Premature defoliation'
        ],
        'causes': 'Fungal infection (Cercospora zeae-maydis or Acceratherum zeae)',
        'solutions': [
            'Use resistant corn hybrids',
            'Apply fungicides (azoxystrobin, pyraclostrobin) at first sign',
            'Practice crop rotation (3-4 years)',
            'Remove infected plant debris',
            'Maintain good air circulation',
            'Avoid overhead irrigation',
            'Apply foliar fungicides at V6-V8 growth stage'
        ],
        'prevention': 'Crop rotation, resistant varieties, proper field sanitation, avoid residue in field'
    },
    
    'Corn_(maize)___Common_rust_': {
        'disease_name': 'Common Rust',
        'affected_crop': 'Corn (maize)',
        'severity': 'Medium',
        'symptoms': [
            'Small, round, reddish-brown pustules on leaves',
            'Pustules on both leaf surfaces',
            'Yellow halos around pustules',
            'Pustules develop clusters',
            'Leaves may dry and shred'
        ],
        'causes': 'Fungal infection (Puccinia sorghi)',
        'solutions': [
            'Plant resistant corn hybrids',
            'Apply fungicides (triazoles) when pustules appear',
            'Scout fields regularly for early detection',
            'Remove volunteer corn plants',
            'Avoid planting susceptible varieties',
            'Apply preventive fungicides in high-pressure areas'
        ],
        'prevention': 'Use resistant hybrids, crop rotation, remove alternative hosts, early detection'
    },
    
    'Corn_(maize)___Northern_Leaf_Blight': {
        'disease_name': 'Northern Leaf Blight',
        'affected_crop': 'Corn (maize)',
        'severity': 'High',
        'symptoms': [
            'Long, narrow, tan lesions with rounded ends',
            'Dark borders on lesions',
            'Lesions on lower leaves first',
            'Spore-bearing structures on lesion undersides',
            'Leaf death and premature defoliation'
        ],
        'causes': 'Fungal infection (Exserohilum turcicum)',
        'solutions': [
            'Plant resistant corn hybrids',
            'Apply fungicides (azoxystrobin, propiconazole) at V8-V10',
            'Rotate crops (2-3 year interval)',
            'Remove crop residue from field',
            'Avoid overhead irrigation',
            'Monitor fields regularly',
            'Apply additional fungicide if weather favors disease'
        ],
        'prevention': 'Resistant varieties, crop rotation, field sanitation, no-till planting'
    },
    
    'Corn_(maize)___healthy': {
        'disease_name': 'Healthy Corn',
        'affected_crop': 'Corn (maize)',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Maintain consistent irrigation (1-1.5 inches per week)',
            'Apply balanced fertilizer (apply nitrogen in splits)',
            'Monitor for pests (European corn borer, armyworm)',
            'Scout fields weekly for disease signs',
            'Ensure proper plant spacing'
        ],
        'prevention': 'Regular monitoring, proper nutrition, adequate water, pest management'
    },
    
    'Grape___Black_rot': {
        'disease_name': 'Black Rot',
        'affected_crop': 'Grape',
        'severity': 'High',
        'symptoms': [
            'Dark lesions on leaves with red/brown halos',
            'Dark, shriveled fruit (bird\'s eye appearance)',
            'Fruit becomes hard and mummified',
            'Black pycnidia visible on lesions',
            'Stem and tendril lesions'
        ],
        'causes': 'Fungal infection (Guignardia bidwellii)',
        'solutions': [
            'Remove and destroy infected fruit and leaves',
            'Apply fungicides (mancozeb, sulfur) every 10-14 days',
            'Start spraying at bud break',
            'Prune for better air circulation',
            'Train vines to improve light exposure',
            'Remove fallen leaves and debris',
            'Use resistant rootstocks when possible'
        ],
        'prevention': 'Sanitation, fungicide program, proper pruning, resistant varieties'
    },
    
    'Grape___Esca_(Black_Measles)': {
        'disease_name': 'Esca (Black Measles)',
        'affected_crop': 'Grape',
        'severity': 'High',
        'symptoms': [
            'Reddish-brown spots on white berries',
            'Interveinal red blotching on leaves',
            'Leaves look mottled or striped',
            'Leaf edges may curl and turn brown',
            'Berries shrivel and drop',
            'Wood shows dark streaking when cut'
        ],
        'causes': 'Complex fungal infection (Phaeoacremonium and Fomitiporia species)',
        'solutions': [
            'Remove and destroy infected vines if severely affected',
            'Prune out diseased wood',
            'Seal large pruning cuts with wound dressing',
            'Apply copper fungicides',
            'Improve vine vigor with proper nutrition',
            'Ensure good drainage and air circulation',
            'Remove severely affected vines'
        ],
        'prevention': 'Avoid pruning wounds, proper wound care, reduce stress, phytosanitary measures'
    },
    
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'disease_name': 'Leaf Blight (Isariopsis Leaf Spot)',
        'affected_crop': 'Grape',
        'severity': 'Medium',
        'symptoms': [
            'Brown lesions with red or purple borders',
            'Lesions have concentric rings',
            'Veins may be discolored',
            'Leaves yellow and drop',
            'Can affect berries causing spots'
        ],
        'causes': 'Fungal infection (Isariopsis clavispora)',
        'solutions': [
            'Apply fungicides (mancozeb, sulfur) preventively',
            'Remove infected leaves promptly',
            'Improve air circulation through pruning',
            'Avoid overhead watering',
            'Remove fallen leaves and debris',
            'Spray at bud break and continue at 10-14 day intervals',
            'Use resistant varieties if available'
        ],
        'prevention': 'Good sanitation, fungicide program, air circulation, resistant varieties'
    },
    
    'Grape___healthy': {
        'disease_name': 'Healthy Grape',
        'affected_crop': 'Grape',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water deeply 2-3 times per week',
            'Apply balanced fertilizer (10-10-10) in spring',
            'Prune in winter for shape and productivity',
            'Thin fruit for better quality',
            'Monitor for pests and diseases',
            'Provide proper trellis support'
        ],
        'prevention': 'Proper care, monitoring, adequate nutrition'
    },
    
    'Orange___Haunglongbing_(Citrus_greening)': {
        'disease_name': 'Huanglongbing (Citrus Greening)',
        'affected_crop': 'Orange',
        'severity': 'Critical',
        'symptoms': [
            'Yellowing of leaves (asymmetrical)',
            'Blotchy mottle on leaves',
            'Misshapen, small fruit',
            'Fruit with thick, pale skin',
            'Bitter, acidic juice with green color',
            'Stunted tree growth',
            'Branch dieback'
        ],
        'causes': 'Bacterial infection (Candidatus Liberibacter asiaticus) spread by psyllids',
        'solutions': [
            'No cure - removal and destruction of infected trees is primary control',
            'Control Asian citrus psyllid with insecticides',
            'Avoid moving infected plant material',
            'Use disease-free nursery stock',
            'Notify agricultural authorities',
            'Implement quarantine measures',
            'Plant tolerant/resistant rootstocks in new orchards'
        ],
        'prevention': 'Psyllid control, use disease-free stock, quarantine, regulatory compliance'
    },
    
    'Peach___Bacterial_spot': {
        'disease_name': 'Bacterial Spot',
        'affected_crop': 'Peach',
        'severity': 'Medium',
        'symptoms': [
            'Small, brown, angular spots on leaves',
            'Red halos around leaf spots',
            'Spots on fruit appear as shallow, brown lesions',
            'Fruit spots may have raised edges',
            'Leaves yellow and drop',
            'Fruit becomes unmarketable'
        ],
        'causes': 'Bacterial infection (Xanthomonas species)',
        'solutions': [
            'Apply copper fungicides (fixed copper) preventively',
            'Prune infected branches',
            'Remove and destroy severely infected leaves',
            'Improve air circulation',
            'Avoid overhead watering',
            'Use resistant peach varieties',
            'Sanitize pruning tools'
        ],
        'prevention': 'Use resistant varieties, copper sprays, proper spacing, avoid overhead watering'
    },
    
    'Peach___healthy': {
        'disease_name': 'Healthy Peach',
        'affected_crop': 'Peach',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water deeply 1-2 inches per week',
            'Apply balanced fertilizer (8-8-8) in early spring',
            'Prune annually for shape and air circulation',
            'Thin fruit to 4-6 inches apart',
            'Monitor for pests and diseases',
            'Apply netting to protect from birds if needed'
        ],
        'prevention': 'Proper cultural practices, monitoring, pest management'
    },
    
    'Pepper,_bell___Bacterial_spot': {
        'disease_name': 'Bacterial Spot on Bell Pepper',
        'affected_crop': 'Pepper, bell',
        'severity': 'Medium',
        'symptoms': [
            'Small, dark, greasy-looking spots on leaves',
            'Spots have yellow halos',
            'Spots start from lower leaves',
            'Similar spots on fruit',
            'Premature defoliation',
            'Fruit becomes unmarketable'
        ],
        'causes': 'Bacterial infection (Xanthomonas species)',
        'solutions': [
            'Apply copper fungicides at first sign of disease',
            'Use disease-free seeds and seedlings',
            'Avoid overhead watering',
            'Remove infected leaves and plants',
            'Improve air circulation',
            'Use resistant pepper varieties',
            'Rotate crops (3-year interval)'
        ],
        'prevention': 'Disease-free seed/transplants, resistant varieties, crop rotation, overhead watering avoidance'
    },
    
    'Pepper,_bell___healthy': {
        'disease_name': 'Healthy Bell Pepper',
        'affected_crop': 'Pepper, bell',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water consistently (1-2 inches per week)',
            'Apply balanced fertilizer (10-10-10)',
            'Use mulch to retain moisture',
            'Provide support with stakes or cages',
            'Monitor for pests',
            'Scout for early disease signs'
        ],
        'prevention': 'Proper watering, nutrition, support, monitoring'
    },
    
    'Potato___Early_blight': {
        'disease_name': 'Early Blight',
        'affected_crop': 'Potato',
        'severity': 'High',
        'symptoms': [
            'Brown, circular lesions with concentric rings on lower leaves',
            'Yellow halos around lesions',
            'Lesions enlarge and coalesce',
            'Lower leaves yellow and die',
            'Spots on stems and tubers',
            'Tuber spots have dark, corky appearance'
        ],
        'causes': 'Fungal infection (Alternaria solani)',
        'solutions': [
            'Plant resistant potato varieties',
            'Apply fungicides (mancozeb, chlorothalonil) every 7-10 days',
            'Remove infected leaves and crop residue',
            'Avoid overhead watering',
            'Improve air circulation',
            'Destroy infected tubers',
            'Rotate crops (3-year interval)'
        ],
        'prevention': 'Resistant varieties, crop rotation, fungicide program, field sanitation'
    },
    
    'Potato___Late_blight': {
        'disease_name': 'Late Blight',
        'affected_crop': 'Potato',
        'severity': 'Critical',
        'symptoms': [
            'Water-soaked spots on leaves',
            'Spots have white powdery coating (spores) on undersides',
            'Spots brown and decay quickly',
            'Brown, soft rot on tubers',
            'Rapid plant collapse in wet conditions',
            'Musty odor from infected tissue'
        ],
        'causes': 'Oomycete (Phytophthora infestans)',
        'solutions': [
            'Plant resistant potato varieties',
            'Apply fungicides (mancozeb, metalaxyl-m) preventively every 7-10 days',
            'Avoid overhead watering',
            'Improve air circulation',
            'Remove infected plants immediately',
            'Destroy infected tubers',
            'Apply fungicides if disease detected'
        ],
        'prevention': 'Resistant varieties, fungicide program, moisture management, resistant seed'
    },
    
    'Potato___healthy': {
        'disease_name': 'Healthy Potato',
        'affected_crop': 'Potato',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water consistently (1-2 inches per week)',
            'Apply balanced fertilizer (5-10-10)',
            'Hill soil around plants as they grow',
            'Monitor for pests (Colorado beetle, aphids)',
            'Scout for disease signs regularly',
            'Maintain proper spacing'
        ],
        'prevention': 'Proper care, monitoring, pest management'
    },
    
    'Raspberry___healthy': {
        'disease_name': 'Healthy Raspberry',
        'affected_crop': 'Raspberry',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water deeply 1-2 inches per week',
            'Apply balanced fertilizer (10-10-10)',
            'Prune old canes after fruiting',
            'Provide trellis support',
            'Thin canes for air circulation',
            'Monitor for pests and diseases'
        ],
        'prevention': 'Proper pruning, watering, nutrition, monitoring'
    },
    
    'Soybean___healthy': {
        'disease_name': 'Healthy Soybean',
        'affected_crop': 'Soybean',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water consistently (1.5-2 inches per week during pod fill)',
            'Apply balanced fertilizer if needed',
            'Monitor for pests (beetles, caterpillars, aphids)',
            'Scout for disease signs regularly',
            'Maintain proper row spacing',
            'Time harvest for optimal moisture content'
        ],
        'prevention': 'Monitoring, pest management, adequate water, proper nutrition'
    },
    
    'Squash___Powdery_mildew': {
        'disease_name': 'Powdery Mildew on Squash',
        'affected_crop': 'Squash',
        'severity': 'Medium',
        'symptoms': [
            'White, powdery coating on leaves',
            'Coating appears on both leaf surfaces',
            'Affected leaves curl and dry',
            'Reduced photosynthesis',
            'May affect fruit quality',
            'Premature leaf senescence'
        ],
        'causes': 'Fungal infection (Podosphaera xanthii)',
        'solutions': [
            'Apply sulfur fungicides (avoid when temps exceed 85°F)',
            'Use potassium bicarbonate sprays',
            'Apply neem oil sprays',
            'Remove heavily infected leaves',
            'Improve air circulation by pruning',
            'Avoid overhead watering',
            'Use resistant squash varieties'
        ],
        'prevention': 'Resistant varieties, proper spacing, air circulation, fungicide program'
    },
    
    'Strawberry___Leaf_scorch': {
        'disease_name': 'Leaf Scorch',
        'affected_crop': 'Strawberry',
        'severity': 'Medium',
        'symptoms': [
            'Small, dark lesions on leaves',
            'Lesions have tan centers with dark borders',
            'Lesions expand and coalesce',
            'Leaves appear scorched and brown',
            'Premature defoliation',
            'Reduced fruit production'
        ],
        'causes': 'Fungal infection (Diplocarpon maculatum)',
        'solutions': [
            'Apply fungicides (mancozeb, sulfur) every 7-10 days',
            'Remove infected leaves regularly',
            'Remove runners and old foliage after harvest',
            'Improve air circulation',
            'Avoid overhead watering',
            'Use disease-free transplants',
            'Rotate fields (3-5 year interval)'
        ],
        'prevention': 'Disease-free plants, crop rotation, fungicide program, air circulation'
    },
    
    'Strawberry___healthy': {
        'disease_name': 'Healthy Strawberry',
        'affected_crop': 'Strawberry',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water consistently (1-2 inches per week)',
            'Apply balanced fertilizer throughout growing season',
            'Mulch around plants to reduce disease',
            'Remove runners to direct energy to fruit',
            'Monitor for pests and diseases',
            'Replace plants every 3-4 years'
        ],
        'prevention': 'Proper cultural practices, monitoring, regular plant replacement'
    },
    
    'Tomato___Bacterial_spot': {
        'disease_name': 'Bacterial Spot on Tomato',
        'affected_crop': 'Tomato',
        'severity': 'Medium',
        'symptoms': [
            'Small, dark, water-soaked spots on leaves',
            'Yellow halos around leaf spots',
            'Spots on fruit appear as brown, raised lesions',
            'Fruit spots have corky, scabby appearance',
            'Premature leaf drop',
            'Reduced fruit quality'
        ],
        'causes': 'Bacterial infection (Xanthomonas species)',
        'solutions': [
            'Use disease-free seeds and transplants',
            'Apply copper fungicides at first sign',
            'Avoid overhead watering',
            'Remove infected leaves and plants',
            'Improve air circulation',
            'Use resistant tomato varieties',
            'Rotate crops (3-year minimum)'
        ],
        'prevention': 'Disease-free seed/transplants, resistant varieties, crop rotation, copper sprays'
    },
    
    'Tomato___Early_blight': {
        'disease_name': 'Early Blight on Tomato',
        'affected_crop': 'Tomato',
        'severity': 'High',
        'symptoms': [
            'Brown, circular lesions with concentric rings',
            'Lesions appear on lower leaves first',
            'Yellow halos around lesions',
            'Spots coalesce causing larger necrotic areas',
            'Leaves yellow and drop',
            'Stem cankers may form'
        ],
        'causes': 'Fungal infection (Alternaria solani)',
        'solutions': [
            'Remove infected leaves promptly',
            'Apply fungicides (mancozeb, chlorothalonil) every 7-10 days',
            'Improve air circulation',
            'Avoid overhead watering',
            'Use resistant tomato varieties',
            'Remove and destroy infected plants',
            'Rotate crops (2-3 year interval)'
        ],
        'prevention': 'Resistant varieties, fungicide program, air circulation, crop rotation'
    },
    
    'Tomato___Late_blight': {
        'disease_name': 'Late Blight on Tomato',
        'affected_crop': 'Tomato',
        'severity': 'Critical',
        'symptoms': [
            'Water-soaked spots on leaves and stems',
            'White spore layer on leaf undersides',
            'Spots expand rapidly in wet conditions',
            'Fruit develops brown, sunken lesions',
            'Rapid plant collapse',
            'Musty odor from infected tissue'
        ],
        'causes': 'Oomycete (Phytophthora infestans)',
        'solutions': [
            'Apply fungicides (mancozeb, metalaxyl-m) preventively',
            'Remove infected plants immediately',
            'Avoid overhead watering',
            'Improve air circulation',
            'Use resistant tomato varieties',
            'Destroy infected fruit and foliage',
            'Rotate crops'
        ],
        'prevention': 'Resistant varieties, fungicide program, moisture management, field sanitation'
    },
    
    'Tomato___Leaf_Mold': {
        'disease_name': 'Leaf Mold',
        'affected_crop': 'Tomato',
        'severity': 'Medium',
        'symptoms': [
            'Yellowish spots on upper leaf surface',
            'Gray-brown mold on leaf undersides',
            'Spots enlarge and coalesce',
            'Leaves yellow and drop',
            'Occurs mostly on lower leaves',
            'Worse in high humidity'
        ],
        'causes': 'Fungal infection (Passalora fulva or Cladosporium fulvum)',
        'solutions': [
            'Improve air circulation by pruning',
            'Reduce humidity (avoid overhead watering)',
            'Apply fungicides (sulfur, mancozeb) if needed',
            'Remove infected leaves',
            'Use resistant tomato varieties',
            'Ensure proper spacing between plants',
            'Monitor humidity levels'
        ],
        'prevention': 'Resistant varieties, good air circulation, proper spacing, humidity management'
    },
    
    'Tomato___Septoria_leaf_spot': {
        'disease_name': 'Septoria Leaf Spot',
        'affected_crop': 'Tomato',
        'severity': 'Medium',
        'symptoms': [
            'Circular, brown lesions with concentric rings',
            'Lesions have dark borders',
            'Gray center with black pycnidia',
            'Lesions appear on lower leaves first',
            'Yellow halos around lesions',
            'Premature defoliation'
        ],
        'causes': 'Fungal infection (Septoria lycopersici)',
        'solutions': [
            'Remove infected leaves promptly',
            'Apply fungicides (mancozeb, chlorothalonil) every 7-10 days',
            'Improve air circulation',
            'Avoid overhead watering',
            'Use disease-free seeds',
            'Remove and destroy infected plants',
            'Rotate crops (2-3 year interval)'
        ],
        'prevention': 'Disease-free seed, fungicide program, air circulation, crop rotation'
    },
    
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'disease_name': 'Spider Mites (Two-spotted)',
        'affected_crop': 'Tomato',
        'severity': 'Medium',
        'symptoms': [
            'Fine webbing on leaves',
            'Yellow or brown stippling on leaves',
            'Leaves appear bleached',
            'Wilting despite adequate moisture',
            'Visible mites on plant (magnification needed)',
            'Mites congregate on leaf undersides'
        ],
        'causes': 'Mite infestation (Tetranychus urticae)',
        'solutions': [
            'Spray with water to dislodge mites',
            'Apply miticides (sulfur, neem oil)',
            'Use predatory mites (Phytoseiulus persimilis)',
            'Improve humidity to discourage mites',
            'Remove heavily infested leaves',
            'Apply insecticidal soap',
            'Monitor plants regularly'
        ],
        'prevention': 'Monitor plants, maintain humidity, avoid excessive nitrogen, regular scouting'
    },
    
    'Tomato___Target_Spot': {
        'disease_name': 'Target Spot',
        'affected_crop': 'Tomato',
        'severity': 'Medium',
        'symptoms': [
            'Circular lesions with concentric rings (target-like)',
            'Brown lesions with light centers',
            'Spots surrounded by yellow halos',
            'Spots may have dark border',
            'Lesions coalesce on heavily infected leaves',
            'Leaves yellow and drop'
        ],
        'causes': 'Fungal infection (Longidone menisporoides)',
        'solutions': [
            'Apply fungicides (mancozeb, chlorothalonil, copper) every 7-10 days',
            'Remove infected leaves',
            'Improve air circulation',
            'Avoid overhead watering',
            'Remove and destroy infected plants',
            'Use resistant tomato varieties',
            'Rotate crops'
        ],
        'prevention': 'Resistant varieties, fungicide program, air circulation, crop rotation'
    },
    
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'disease_name': 'Tomato Yellow Leaf Curl Virus (TYLCV)',
        'affected_crop': 'Tomato',
        'severity': 'High',
        'symptoms': [
            'Upward curling of leaflets',
            'Yellow coloration of leaves',
            'Leaflets become thick and brittle',
            'Stunted plant growth',
            'Reduced fruit set and quality',
            'Severe in warm conditions'
        ],
        'causes': 'Viral infection spread by whiteflies',
        'solutions': [
            'Use insecticides to control whiteflies',
            'Remove and destroy infected plants',
            'Install reflective mulch to repel whiteflies',
            'Use row covers on young plants',
            'Plant resistant tomato varieties',
            'Spray with neem oil or insecticidal soap',
            'Control alternative hosts (weeds)'
        ],
        'prevention': 'Virus-free transplants, whitefly control, resistant varieties, row covers'
    },
    
    'Tomato___Tomato_mosaic_virus': {
        'disease_name': 'Tomato Mosaic Virus (TMV)',
        'affected_crop': 'Tomato',
        'severity': 'High',
        'symptoms': [
            'Mottling and mosaic pattern on leaves',
            'Leaf curling and distortion',
            'Stunted plant growth',
            'Reduced fruit set',
            'Fruit may have brown, necrotic patches',
            'Leaf rugosity (wrinkled appearance)'
        ],
        'causes': 'Viral infection (spread by contact, tools, hands)',
        'solutions': [
            'Use disease-free, resistant tomato varieties',
            'Remove and destroy infected plants immediately',
            'Sanitize tools with bleach solution (10%)',
            'Wash hands before handling plants',
            'Avoid handling plants when wet',
            'Control alternative hosts (weeds)',
            'Use virus-free transplants'
        ],
        'prevention': 'Resistant varieties, sanitation, virus-free stock, proper hygiene, resistant cultivars'
    },
    
    'Tomato___healthy': {
        'disease_name': 'Healthy Tomato',
        'affected_crop': 'Tomato',
        'severity': 'None',
        'symptoms': [],
        'causes': 'No disease present',
        'solutions': [
            'Water deeply and consistently (1-2 inches per week)',
            'Apply balanced fertilizer (10-10-10) every 3 weeks',
            'Prune suckers for better air circulation',
            'Provide sturdy support (stakes, cages)',
            'Monitor for pests and diseases weekly',
            'Mulch to retain moisture and suppress weeds'
        ],
        'prevention': 'Proper care, regular monitoring, pest management, adequate nutrition'
    }
}

def disease_detection(request):
    prediction = None
    disease_info = None
    Classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    if request.method == "POST" and request.FILES.get("image"):
        if pytorch_model is None:
            prediction = "Disease detection model is not available. Please contact administrator."
        else:
            image_file = request.FILES["image"]
            prediction = predict_image(image_file, pytorch_model, Classes)
            
            # Get disease information from the dictionary
            if prediction in PLANT_DISEASE_SOLUTIONS:
                disease_info = PLANT_DISEASE_SOLUTIONS[prediction]

    return render(request, "dashboard/disease_detection.html", {
        "prediction": prediction,
        "disease_info": disease_info
    })


def weather_page(request):
    return render(request, 'dashboard/weather_forecast.html')

def weather_forecast(request):
    city = request.GET.get('city', 'London')
    api_key = 'ff7fd28ad0df49a4ab8101120252710'
    url = f'http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=yes'

    try:
        response = requests.get(url)
        data = response.json()
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'error': {'message': str(e)}})


def about(request):
    return render(request, 'dashboard/about.html')

GEMINI_API_KEY = "your_api_key" 
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-flash-lite-latest")
    print("✅ Gemini API initialized successfully")
except Exception as e:
    print(f"❌ Gemini API initialization failed: {e}")
    gemini_model = None


@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            msg = data.get("message", "")

            if not msg:
                return JsonResponse({"reply": "Please type something."})

            try:
                response = gemini_model.generate_content(msg)
                return JsonResponse({"reply": response.text})
            except Exception as e:
                return JsonResponse({"reply": f"Error: {str(e)}"})

        except json.JSONDecodeError:
            return JsonResponse({"reply": "Invalid JSON format."})
        except Exception as e:
            return JsonResponse({"reply": f"Server error: {str(e)}"})

    return JsonResponse({"reply": "Only POST allowed"})

def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        subject = request.POST.get('subject', '')
        message = request.POST.get('message', '')

        context = {
            'success': True,
            'name': name,
            'email': email,
            'subject': subject,
            'message': message,
        }
        return render(request, 'dashboard/contact.html', context)
    
    return render(request, 'dashboard/contact.html')

def government_schemes(request):
    return render(request, 'dashboard/government_schemes.html')

def soil_detection(request):
    prediction = None
    soil_info = None
    Classes: ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil', 'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]
        image_path = os.path.join(settings.MEDIA_ROOT, 'soil_images', image_file.name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        prediction = predict_soil_type(image_path, soil_model)
        
        # Soil type information
        soil_info = get_soil_info(prediction)
        
        # Clean up uploaded file
        try:
            os.remove(image_path)
        except:
            pass
    
    return render(request, "dashboard/soil_detection.html", {
        "prediction": prediction,
        "soil_info": soil_info
    })
