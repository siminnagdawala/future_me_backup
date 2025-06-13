from flask import Flask, request, jsonify, send_file
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import bson
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
import google.generativeai as genai
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import base64
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import re
from PIL import Image as PILImage
from flask_cors import CORS
import markdown
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import traceback



load_dotenv()

app = Flask(__name__)
CORS(app) 

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{2,20}$")

limiter = Limiter(
    get_remote_address,  
    app=app,
    default_limits=["5 per minute"]  
)

MAX_QUERY_LENGTH = 300

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        confirm_password = data.get("confirm_password", "").strip()

        if not USERNAME_REGEX.match(username):
            return jsonify({"message": "Invalid username format"}), 400

        existing_user = mongo.db.users.find_one({"username": username})
        if existing_user:
            return jsonify({"message": "Username already exists"}), 400

        if password != confirm_password:
            return jsonify({"message": "Passwords do not match"}), 400

        hashed_password = generate_password_hash(password)

        user_id = mongo.db.users.insert_one({
            "username": username,
            "password": hashed_password
        }).inserted_id

        return jsonify({"message": "User created", "user_id": str(user_id)}), 201

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()

        if not USERNAME_REGEX.match(username):
            return jsonify({"message": "Invalid username format"}), 400

        user = mongo.db.users.find_one({"username": username})
        if not user or not check_password_hash(user["password"], password):
            return jsonify({"message": "Invalid username or password"}), 401

        prediction_data = mongo.db.predictions.find_one({"username": username}) or {}

        return jsonify({
            "message": "Login successful",
            "prediction": prediction_data.get("career_prediction", ""),
            "riasec_scores": prediction_data.get("riasec_scores", {}),
            "riasec_chart": prediction_data.get("riasec_chart", ""),
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500
# ML Prediction
def predict_career_from_json(input_json):
    model = joblib.load("pkl_files/career_model.pkl")
    label_encoders = joblib.load("pkl_files/label_encoders.pkl")
    target_encoder = joblib.load("pkl_files/target_encoder.pkl")
    df_input = pd.DataFrame([input_json])

    # Encode categorical inputs
    for col in df_input.columns:
        if col in label_encoders:
            df_input[col] = label_encoders[col].transform(df_input[col])

    # Predict
    pred = model.predict(df_input)[0]
    return target_encoder.inverse_transform([pred])[0]

def calculate_riasec_scores(data):
    """Calculate RIASEC scores based on form responses."""
    scores = {
        'Realistic': 2,      
        'Investigative': 2,
        'Artistic': 2,
        'Social': 2,
        'Enterprising': 2,
        'Conventional': 2
    }
    
    performance_map = {
        'Struggle significantly': 2, 
        'Struggle': 3,
        'Average': 4,
        'Good': 5,
        'Excellent': 6
    }
    
    # Academic Performance - reduced multipliers
    scores['Investigative'] += performance_map.get(data['academicPerformance']['mathematics'], 0) * 0.3
    scores['Investigative'] += performance_map.get(data['academicPerformance']['science'], 0) * 0.3
    
    # Favorite Subject mapping - reduced impact
    subject_mappings = {
        'Mathematics': {'Investigative': 1.5, 'Conventional': 1},
        'Science': {'Investigative': 1.5, 'Realistic': 1},
        'Social Studies/History': {'Social': 1.5, 'Investigative': 1},
        'Languages': {'Artistic': 1.5, 'Social': 1},
        'Computer Science/IT': {'Investigative': 1.5, 'Realistic': 1}
    }
    
    if data['academicPerformance']['favoriteSubject'] in subject_mappings:
        for category, value in subject_mappings[data['academicPerformance']['favoriteSubject']].items():
            scores[category] += value

    # Hobbies mapping - smaller increments
    hobby_mappings = {
        'Coding/Technology': {'Investigative': 0.8, 'Realistic': 0.8},
        'Sports/Physical Activities': {'Realistic': 0.8, 'Social': 0.8},
        'Drawing/Painting/Creative Arts': {'Artistic': 1.2},
        'Reading/Writing': {'Artistic': 0.8, 'Investigative': 0.8},
        'Music/Dance/Drama': {'Artistic': 0.8, 'Social': 0.8},
        'Volunteering/Helping Others': {'Social': 1.2}
    }
    
    for hobby in data['interests']['hobbies']:
        if hobby in hobby_mappings:
            for category, value in hobby_mappings[hobby].items():
                scores[category] += value

    # Work Style mapping - reduced impact
    work_style_mappings = {
        'Love group activities and teamwork': {'Social': 1.5, 'Enterprising': 1},
        'Enjoy small groups or one-on-one': {'Social': 1, 'Conventional': 0.5},
        'Prefer working alone': {'Investigative': 1, 'Conventional': 0.8}
    }
    
    style = data['personalityAndWorkStyle']['interactionPreference']
    if style in work_style_mappings:
        for category, value in work_style_mappings[style].items():
            scores[category] += value

    # Decision Making Style - balanced impact
    decision_style = data['personalityAndWorkStyle']['decisionMakingStyle']
    if decision_style == 'Logic and facts':
        scores['Investigative'] += 1.2
        scores['Conventional'] += 0.8
    elif decision_style == 'Emotions and values':
        scores['Social'] += 1.2
        scores['Artistic'] += 0.8

    # Strongest Skill mapping - moderate impact
    skill_mappings = {
        'Problem-solving/Analytical Thinking': {'Investigative': 1.2, 'Realistic': 0.8},
        'Creativity/Artistic Talent': {'Artistic': 1.5},
        'Communication/Leadership': {'Social': 1.2, 'Enterprising': 0.8},
        'Organization/Attention to Detail': {'Conventional': 1.5},
        'Empathy/Teamwork': {'Social': 1.5}
    }
    
    if data['strengthsAndWeaknesses']['strongestSkill'] in skill_mappings:
        for category, value in skill_mappings[data['strengthsAndWeaknesses']['strongestSkill']].items():
            scores[category] += value

    # Normalize scores to 2-8 scale (more balanced range)
    max_score = max(scores.values())
    min_score = min(scores.values())
    
    for category in scores:
        normalized = (scores[category] - min_score) / (max_score - min_score) * 6 + 2  # Scale to 2-8 range
        scores[category] = int(round(normalized))  # Convert to integer

    return scores

def create_riasec_chart(scores):
    """Create RIASEC hexagonal chart with enhanced colors and spacing."""
    categories = list(scores.keys())
    values = list(scores.values())
    
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    
    # Increased figure size and added more padding
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection='polar')
    
    # Enhanced color scheme
    colors = {
        'Realistic': '#FF6B6B',      
        'Investigative': '#4ECDC4',  
        'Artistic': '#45B7D1',       
        'Social': '#96CEB4',         
        'Enterprising': '#FFEEAD',   
        'Conventional': '#D4A5A5'   
    }
    
    # Create gradient effect with straight lines between points
    for i in range(len(categories)):
        ax.fill_between([angles[i], angles[i+1]], [0, 0], [values[i], values[i+1]], 
                       color=colors[categories[i]], alpha=0.25)
    
    # Plot the outline with straight lines
    ax.plot(angles, values, color='#2C3E50', linewidth=2, linestyle='-')
    
    # Add points at each category
    ax.scatter(angles[:-1], values[:-1], c='#2C3E50', s=100)
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    
    # Set radial axes and make grid lines straight
    ax.set_ylim(0, 10)
    ax.set_rgrids([2, 4, 6, 8], angle=0, fontsize=10)
    
    # Make the plot hexagonal by setting the number of grid lines
    ax.set_thetagrids(np.arange(0, 360, 60))
    
    # Straighten the grid lines to create hexagonal shape
    for line in ax.get_xgridlines():
        line.set_linestyle('-')
    
    # Add a title with more padding and bold font
    plt.title('RIASEC Profile', size=16, y=1.08, fontweight='bold')
    
    # Save to BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64


MAX_INPUT_LENGTH = 5000  

@app.route('/predict_career', methods=['POST'])
@limiter.limit("3 per minute") 
def predict_career():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        #   Enforce character limit on input data
        if len(str(data)) > MAX_INPUT_LENGTH:
            return jsonify({"error": f"Input data too long. Max {MAX_INPUT_LENGTH} characters allowed."}), 400

        #   Validate user_id format
        user_id = data["personalInfo"].get("user_id")
        if not bson.ObjectId.is_valid(user_id):
            return jsonify({"error": "Invalid user_id format"}), 400
        
        #   Validate username format
        username = data["personalInfo"].get("username", "").strip()
        if not USERNAME_REGEX.match(username):
            return jsonify({"error": "Invalid username format"}), 400

        print(f"Received Data: {data}")
        
        # try:
        #     _ = predict_career_from_json(data)
        # except Exception:
        #     _ = None  

        prompt = f"""
        You are a career counselor assisting a 10th standard student in selecting a suitable career path. Based on the following information, please recommend the most suitable career path for this student. Additionally, explain why this career is a good fit considering their academic performance, interests, personality, and future goals.

        Academic Performance:
        - Mathematics: {data["academicPerformance"]["mathematics"]}
        - Science: {data["academicPerformance"]["science"]}
        - Social Studies: {data["academicPerformance"]["socialStudies"]}
        - Language: {data["academicPerformance"]["language"]}
        - Favorite Subject: {data["academicPerformance"]["favoriteSubject"]}
        - Least Favorite Subject: {data["academicPerformance"]["leastFavoriteSubject"]}

        Personal Interests:
        - Hobbies: {data["interests"]["hobbies"]}
        - Career Excitement: {data["interests"]["careerExcitement"]}

        Personality and Work Style:
        - Interaction Preference: {data["personalityAndWorkStyle"]["interactionPreference"]}
        - Decision Making Style: {data["personalityAndWorkStyle"]["decisionMakingStyle"]}
        - Deadline Management: {data["personalityAndWorkStyle"]["deadlineManagement"]}
        - Preferred Work Environment: {data["personalityAndWorkStyle"]["workEnvironment"]}

        Strengths and Weaknesses:
        - Strongest Skill: {data["strengthsAndWeaknesses"]["strongestSkill"]}
        - Weakness to Improve: {data["strengthsAndWeaknesses"]["weaknessToImprove"]}
        - Reaction to Challenges: {data["strengthsAndWeaknesses"]["reactionToChallenges"]}

        Career Preferences:
        - Career Research: {data["careerPreferences"]["careerResearch"]}
        - Family Influence: {data["careerPreferences"]["familyInfluence"]}
        - Preferred Work Environment: {data["careerPreferences"]["preferredWorkEnvironment"]}

        Future Goals:
        - Job Stability: {data["futureGoals"]["jobStability"]}
        - Higher Education: {data["futureGoals"]["higherEducation"]}
        - Willingness to Relocate: {data["futureGoals"]["willingnessToRelocate"]}

        Scenario Responses:
        - Winning Money Choice: {data["scenarios"]["winningMoneyChoice"]}
        - Group Project Choice: {data["scenarios"]["groupProjectChoice"]}
        - Learning Style: {data["scenarios"]["learningStyle"]}
        - Role Models: {data["scenarios"]["roleModels"]}
        - Post-10th Plans: {data["scenarios"]["post10thPlans"]}

        Career Path Options:
        1. Animation, Graphics, and Multimedia
        2. Bachelor of Architecture (B.Arch)
        3. Bachelor of Commerce (B.Com)
        4. Bachelor of Education (B.Ed.)
        5. Bachelor of Science in Applied Geology (B.Sc- Applied Geology)
        6. Bachelor of Science in Nursing (B.Sc- Nursing)
        7. Bachelor of Science in Chemistry (B.Sc. Chemistry)
        8. Bachelor of Science in Mathematics (B.Sc. Mathematics)
        9. Bachelor of Science in Information Technology (B.Sc.- IT)
        10. Bachelor of Science in Physics (B.Sc.- Physics)
        11. B.Tech in Civil Engineering
        12. B.Tech in Computer Science and Engineering
        13. B.Tech in Electrical and Electronics Engineering
        14. B.Tech in Electronics and Communication Engineering
        15. B.Tech in Mechanical Engineering
        16. Bachelor of Arts in Economics (BA in Economics)
        17. Bachelor of Arts in English (BA in English)
        18. Bachelor of Arts in Hindi (BA in Hindi)
        19. Bachelor of Arts in History (BA in History)
        20. Bachelor of Business Administration (BBA)
        21. Bachelor of Business Studies (BBS)
        22. Bachelor of Computer Applications (BCA)
        23. Bachelor of Dental Surgery (BDS)
        24. Bachelor of Event Management (BEM)
        25. Bachelor of Fashion Designing (BFD)
        26. Bachelor of Journalism and Mass Communication (BJMC)
        27. Bachelor of Pharmacy (BPharma)
        28. Bachelor of Travel and Tourism Management (BTTM)
        29. Bachelor of Visual Arts (BVA)
        30. Chartered Accountancy (CA)
        31. Company Secretary (CS)
        32. Civil Services
        33. Diploma in Dramatic Arts
        34. Integrated Law Course (BA + LL.B)
        35. Medical (MBBS)

        Please select the most suitable career path from the above options and provide your recommendation in plain text format. Explain why this choice aligns well with the student's strengths, interests, and future goals. 
        """

        response = model.generate_content(prompt)

        if response and hasattr(response, 'text'):
            prediction = response.text

            #  Calculate RIASEC scores and generate a chart
            riasec_scores = calculate_riasec_scores(data)
            riasec_chart = create_riasec_chart(riasec_scores)
        
        # Store in MongoDB
            prediction_id = mongo.db.predictions.insert_one({
                "user_id": ObjectId(data["personalInfo"]["user_id"]),
                "name": data["personalInfo"]["name"],
                "school": data["personalInfo"]["school"],
                "username": data["personalInfo"]["username"],
                
                "mathematics_performance": data["academicPerformance"]["mathematics"],
                "science_performance": data["academicPerformance"]["science"],
                "social_studies_performance": data["academicPerformance"]["socialStudies"],
                "language_performance": data["academicPerformance"]["language"],
                "favorite_subject": data["academicPerformance"]["favoriteSubject"],
                "least_favorite_subject": data["academicPerformance"]["leastFavoriteSubject"],

                "hobbies": data["interests"]["hobbies"],
                "career_excitement": data["interests"]["careerExcitement"],

                "interaction_preference": data["personalityAndWorkStyle"]["interactionPreference"],
                "decision_making_style": data["personalityAndWorkStyle"]["decisionMakingStyle"],
                "deadline_management": data["personalityAndWorkStyle"]["deadlineManagement"],
                "work_environment": data["personalityAndWorkStyle"]["workEnvironment"],

                "strongest_skill": data["strengthsAndWeaknesses"]["strongestSkill"],
                "weakness_to_improve": data["strengthsAndWeaknesses"]["weaknessToImprove"],
                "reaction_to_challenges": data["strengthsAndWeaknesses"]["reactionToChallenges"],

                "career_research": data["careerPreferences"]["careerResearch"],
                "family_influence": data["careerPreferences"]["familyInfluence"],
                "preferred_work_environment": data["careerPreferences"]["preferredWorkEnvironment"],

                "job_stability": data["futureGoals"]["jobStability"],
                "higher_education": data["futureGoals"]["higherEducation"],
                "willingness_to_relocate": data["futureGoals"]["willingnessToRelocate"],

                "winning_money_choice": data["scenarios"]["winningMoneyChoice"],
                "group_project_choice": data["scenarios"]["groupProjectChoice"],
                "learning_style": data["scenarios"]["learningStyle"],
                "role_models": data["scenarios"]["roleModels"],
                "post_10th_plans": data["scenarios"]["post10thPlans"],
                
                "marks": { 
                    "science": data["marks"]["science"],  
                    "english": data["marks"]["english"], 
                    "hindi": data["marks"]["hindi"], 
                    "marathi": data["marks"]["marathi"],  
                    "socialScience": data["marks"]["socialScience"],  
                },
                
                "career_prediction": prediction,
                "riasec_scores": riasec_scores,
                "riasec_chart": riasec_chart
            }).inserted_id
            
            return jsonify({
                    "message": "Data received and stored successfully!",
                    "prediction": prediction,
                    "riasec_scores": riasec_scores,
                    "riasec_chart": riasec_chart
                }), 201

        else:
            return jsonify({"message": "Data received but failed to get prediction from AI."}), 500

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500



@app.route('/generate_pdf/<username>', methods=['GET'])
def generate_pdf(username):
    # Fetch user data from MongoDB using username
    user_data = mongo.db.predictions.find_one({"username": username})
    
    if not user_data:
        return jsonify({"error": "User data not found"}), 404

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30)
    story.append(Paragraph("Career Assessment Report", title_style))
    
    # Personal Information
    story.append(Paragraph("Personal Information", styles['Heading2']))
    personal_details = [
        f"Name: {user_data.get('name', 'N/A')}",
        f"School: {user_data.get('school', 'N/A')}"
    ]
    for info in personal_details:
        story.append(Paragraph(info, styles['Normal']))
    story.append(Spacer(1, 20))

    # Academic Performance
    story.append(Paragraph("Academic Performance", styles['Heading2']))
    marks = user_data.get('marks', {})
    for subject, score in marks.items():
        story.append(Paragraph(f"{subject.capitalize()}: {score}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Extract and display all other sections correctly
    sections = {
        "Interests and Hobbies": ['hobbies', 'career_excitement'],
        "Personality & Work Style": ['interaction_preference', 'decision_making_style', 'deadline_management', 'work_environment'],
        "Strengths & Weaknesses": ['strongest_skill', 'weakness_to_improve', 'reaction_to_challenges'],
        "Career Preferences": ['career_research', 'family_influence', 'preferred_work_environment'],
        "Future Goals": ['job_stability', 'higher_education', 'willingness_to_relocate']
    }

    for section, keys in sections.items():
        story.append(Paragraph(section, styles['Heading2']))
        for key in keys:
            value = user_data.get(key, 'N/A')
            if key == "hobbies" and isinstance(value, list):
                value = ', '.join(map(str, value))  # Convert list to a comma-separated string
            story.append(Paragraph(f"{key.replace('_', ' ').capitalize()}: {value}", styles['Normal']))
        story.append(Spacer(1, 20))

    # RIASEC Chart (Maintaining Aspect Ratio)
    if user_data.get('riasec_chart'):
        story.append(Paragraph("RIASEC Profile", styles['Heading2']))
        
        chart_data = base64.b64decode(user_data['riasec_chart'])
        img_buffer = BytesIO(chart_data)
        
        pil_img = PILImage.open(img_buffer)
        orig_width, orig_height = pil_img.size
        new_width = 300
        aspect_ratio = orig_height / orig_width
        new_height = int(new_width * aspect_ratio)

        img_buffer.seek(0)
        img = Image(img_buffer, width=new_width, height=new_height)
        story.append(img)
    
    story.append(Spacer(1, 20))

    # Career Prediction with Markdown Formatting (Fixing new lines and bullets)
    story.append(Paragraph("Career Prediction", styles['Heading2']))
    career_prediction = user_data.get('career_prediction', 'No prediction available')

    def markdown_to_pdf_text(md_text):
        md_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', md_text)
        md_text = re.sub(r'\*(.*?)\*', r'<b>\1</b>', md_text)
        md_text = re.sub(r'#+\s*(.*?)', r'<b>\1</b>', md_text)
        md_text = md_text.replace("\n", "<br/>")
        md_text = re.sub(r'^\s*-\s*(.*?)$', r'• \1', md_text, flags=re.MULTILINE)

        return md_text

    formatted_text = markdown_to_pdf_text(career_prediction)
    story.append(Paragraph(formatted_text, styles['Normal']))

    # Build PDF
    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"career_assessment_{user_data.get('name', 'user')}.pdf",
        mimetype='application/pdf'
    )


def get_user_data(username):
    user = mongo.db.predictions.find_one({"username": username})
    if user:
        return {
            "career_prediction": user.get("career_prediction", "Not Available"),
            "interests": user.get("hobbies", []),
            "strengths": user.get("strongest_skill", ""),
            "academicPerformance": {
                "math": user.get("mathematics_performance", ""),
                "science": user.get("science_performance", ""),
                "socialStudies": user.get("social_studies_performance", ""),
                "language": user.get("language_performance", ""),
                "favoriteSubject": user.get("favorite_subject", ""),
                "leastFavoriteSubject": user.get("least_favorite_subject", ""),
            }
        }
    return {}

# Function to generate AI response
def generate_response(query, user_data):
    #  Enforcing AI to provide structured & factual answers
    prompt = f"""
    You are a **career guidance chatbot** for **10th-grade students in India**.
    Your goal is to **provide accurate, reliable, and structured career advice** based on official guidelines.

    **Student Background:**
    - **Predicted Career:** {user_data.get('career_prediction', 'Not Available')}
    - **Interests:** {', '.join(user_data.get('interests', [])) if user_data.get('interests') else 'Not Available'}
    - **Strengths:** {', '.join(user_data.get('strengths', [])) if user_data.get('strengths') else 'Not Available'}

    **Rules for Answering:**
    - **Only answer questions related to career, education, subjects, and future goals.**
    - ** Answer to any greetings or career worries and doubts"**
    - **If a career path is mentioned, give proper step-by-step guidance.**
    - **Cite indian official sources whenever possible (like NCERT, NIRF, Govt Websites, etc).**
    - **If you don't know, say: 'I am unsure. Please check official websites and provide them relevant official websites for that particular career.'**
    
    **Query:** "{query}"
    
    **Now, provide a well-structured answer in Markdown format.**
    """

    response = model.generate_content(prompt)

    markdown_text = response.text.strip()

    # Post-processing: Detect unreliable answers
    if "I don't know" in markdown_text or "unsure" in markdown_text:
        return "⚠️ I am unsure. Please check official sources like [NCERT](https://ncert.nic.in/) or [NIRF](https://www.nirfindia.org/)."

    # Convert Markdown response to HTML (for debugging)
    html_text = markdown.markdown(markdown_text)

    return markdown_text  # Return cleaned Markdown response

@app.route("/chat", methods=["POST"])
@limiter.limit("5 per minute")  #  Rate limit applied
def chat():
    try:
        data = request.json
        username = data.get("username", "").strip()
        query = data.get("query", "").strip()

        if not USERNAME_REGEX.match(username):
            return jsonify({"error": "Invalid username format"}), 400

        if not username or not query:
            return jsonify({"error": "Missing username or query"}), 400

        if len(query) > MAX_QUERY_LENGTH:
            return jsonify({"error": f"Query too long. Max {MAX_QUERY_LENGTH} characters allowed."}), 400

        # Fetch user data securely
        user_data = get_user_data(username)
        ai_response = generate_response(query, user_data)

        # Save chat securely
        mongo.db.chat_history.insert_one({
            "username": username,
            "query": query,
            "response": ai_response
        })

        return jsonify({"query": query, "response": ai_response})

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/chat/history", methods=["GET"])
@limiter.limit("5 per minute")  
def get_chat_history():
    try:
        username = request.args.get("username", "").strip()

        # Validate username format
        if not USERNAME_REGEX.match(username):
            return jsonify({"error": "Invalid username format"}), 400

        if not username:
            return jsonify({"error": "Missing username"}), 400

        # Fetch chat history securely
        history = list(mongo.db.chat_history.find(
            {"username": username}, {"_id": 0}
        ))

        return jsonify({"history": history})

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Backend is running!",
        "port": os.environ.get("PORT", 5000),
        "host": "0.0.0.0"
    })

print(f"DEBUG: PORT environment variable = {os.environ.get('PORT', 'NOT SET')}")
print(f"DEBUG: All environment variables: {list(os.environ.keys())}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running on port: {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
