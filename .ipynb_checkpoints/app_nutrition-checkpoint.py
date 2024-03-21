import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64

gym_data = pd.read_csv('gym_data.csv')

def create_strength_table():
    strength_table = pd.read_csv("strength_table.csv")
    return strength_table

def load_recipes_data():
    # Load recipes dataset from CSV file
    recipes_data = pd.read_csv('recipes_data.csv')
    return recipes_data

# Function to load and preprocess data
def load_and_preprocess_data():
    # Load dataset from file in the same folder as the script
    data = pd.read_csv('numeric_data.csv')
    
    # Train the model
    X = data.drop(['calories_to_maintain_weight'], axis=1)  
    y = data['calories_to_maintain_weight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def load_food_data():
    # Load the food dataset
    food_data = pd.read_csv("food_clean.csv")
    return food_data

def find_similar_food(food_data, input_food):
    # Filter foods containing the keyword
    similar_foods = food_data[food_data['name'].str.contains(input_food, case=False)]
    return similar_foods

def display_nutrient_summary(food_data):
    # Show nutrient summary for each food
    st.write(food_data)
    for index, row in food_data.iterrows():
        plot_nutrients_distribution(food_data.loc[[index]], row['name'])

import plotly.graph_objects as go

def plot_nutrients_distribution(food_data, food_name):
    # Filter the first 3 foods containing the specified name
    filtered_food_data = food_data[food_data['name'].str.contains(food_name, case=False)].head(3)

    # Check if foods were found
    if not filtered_food_data.empty:
        # Get nutrients
        nutrients = ['calories', 'carbs_g', 'fat_g', 'protein_g']

        # Create subplots for each food
        fig = go.Figure()

        for index, row in filtered_food_data.iterrows():
            # Sort nutrient values from highest to lowest
            sorted_values = row[nutrients].sort_values(ascending=False)
            # Get sorted nutrient names
            sorted_nutrients = sorted_values.index.tolist()
            # Get sorted values
            sorted_values = sorted_values.tolist()

            # Create donut chart
            fig.add_trace(go.Pie(labels=sorted_nutrients, values=sorted_values, hole=0.3, name=row['name']))

        # Set layout
        fig.update_layout(title_text=f'Macros Chart for: "{food_name}"',
                          template='plotly_dark')

        # Show the chart
        st.plotly_chart(fig)

    else:
        st.write(f"No foods containing '{food_name}' were found.")


def display_calories_info(recipe_data):
    st.write(f"Calories: {recipe_data['calories'].values[0]}")

def plot_nutrient_distribution(recipe_data):
    # Get values of requested columns
    nutrients = recipe_data[['total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 'protein_pdv', 'saturated_fat', 'carbohydrates']]

    # Get nutrient labels
    labels = nutrients.columns

    # Get nutrient values
    values = nutrients.iloc[0].values

    # Create donut chart figure
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

    # Add title
    fig.update_layout(title_text='Nutrient Distribution')

    # Set layout theme
    fig.update_layout(template='plotly_dark')

    # Show the chart
    st.plotly_chart(fig)

def display_recipe_info(recipe_data):
    st.write(f"Recipe: {recipe_data['name']}")
    st.write("Steps to make the recipe:")
    steps = eval(recipe_data['steps'])
    for step in steps:
        st.write(step)

    st.write("\nAdditional recipe information:")
    st.write(f"Preparation time: {recipe_data['minutes']} minutes")
    st.write("Ingredients:")
    ingredients = eval(recipe_data['ingredients'])
    for ingredient in ingredients:
        st.write(f"- {ingredient}")

def generate_recipe_ideas(recipes_data, available_ingredients, num_recipes=5):
    # Convert ingredients to lowercase for easier comparison
    available_ingredients = [ingredient.strip().lower() for ingredient in available_ingredients]

    # Filter recipes containing at least one available ingredient
    filtered_recipes = recipes_data[recipes_data['ingredients'].apply(lambda x: any(item in x for item in available_ingredients))]

    # Count the number of matching ingredients in each recipe
    filtered_recipes['matching_ingredients'] = filtered_recipes['ingredients'].apply(lambda x: len(set(x) & set(available_ingredients)))

    # Select recipes with the most matching ingredients
    selected_recipes = filtered_recipes.sort_values(by='matching_ingredients', ascending=False).head(num_recipes)

    return selected_recipes

def display_recipe_ideas(selected_recipes):
    if selected_recipes.empty:
        st.write("No recipes found with the available ingredients.")
    else:
        st.write("Here are some recipes you can make with the ingredients you have:")
        for index, row in selected_recipes.iterrows():
            # Show the recipe name as it is in the available recipe list
            st.write(f"- **{row['name']}**: {row['description']}")

def generate_recipe_ideas_section(recipes_data):
    st.title("Recipe Ideas")
    
    # Add image at the top
    st.image("https://restaurantecasaantonio.net/wp-content/uploads/2019/03/Qu%C3%A9-ingredientes-no-pueden-faltar-en-nuestras-recetas.jpg", use_column_width=True)

    # Get user's available ingredients
    available_ingredients = st.text_input("Enter the ingredients you have available, separated by comma:")

    if available_ingredients:
        # Convert ingredients to a list
        available_ingredients = [ingredient.strip() for ingredient in available_ingredients.split(',')]

        # Generate recipe ideas based on available ingredients
        selected_recipes = generate_recipe_ideas(recipes_data, available_ingredients)

        # Show selected recipes
        display_recipe_ideas(selected_recipes)

        # Button to generate 5 more recipes
        if st.button("Generate 5 more recipes"):
            # Get all recipes containing at least one available ingredient
            filtered_recipes = recipes_data[recipes_data['ingredients'].apply(lambda x: any(item in x for item in available_ingredients))]
            
            # Randomly select 5 different recipes
            random_recipes = filtered_recipes.sample(n=5, replace=False)

            # Show the new recipes
            display_recipe_ideas(random_recipes)

        # Ask the user to choose a recipe to view more details
        chosen_recipe = st.text_input("Which recipe would you like to make? Please write the recipe name:")

        if chosen_recipe:
            # Look up detailed information of the chosen recipe
            recipe_data = recipes_data[recipes_data['name'] == chosen_recipe]
            if not recipe_data.empty:
                display_recipe_info(recipe_data.iloc[0])
            else:
                st.write("The specified recipe was not found in the available recipe list.")

def generate_training_plan(gym_data, days_per_week=7):
    training_plan = {}

    # Convert column names to snake case
    gym_data.columns = gym_data.columns.str.lower().str.replace(' ', '_')

    # Ask user how many days a week they want to train
    days_per_week = st.number_input("How many days a week do you want to train?", min_value=1, max_value=7, value=7, step=1)

    for day in range(1, days_per_week + 1):
        st.subheader(f'Day {day}:')
        body_part_preference = st.selectbox(f"What body part do you prefer to train on Day {day}?", ('Upper body', 'Lower body', 'Core', 'Back', 'Full body'))

        if "train" in body_part_preference.lower():
            # If preference contains "train", remove spaces and replace with "_"
            body_part_preference = body_part_preference.lower().replace(" ", "_")
        
        if body_part_preference.lower() == "full body":
            # If user is unsure, generate exercises for all body parts
            selected_exercises = select_exercises_all(gym_data)
        else:
            # Select exercises based on user preferences
            selected_exercises = select_exercises(gym_data, body_part_preference.lower())

        # Display selected exercises in the app
        st.table(selected_exercises[['title', 'type', 'bodypart', 'equipment']])

        # Download table as Excel
        file_name = f"training_plan_day_{day}.xlsx"
        st.markdown(get_table_download_link(selected_exercises[['title', 'type', 'bodypart', 'equipment']], file_name), unsafe_allow_html=True)
        
def select_exercises(gym_data, body_part_preference):
    # Translate user preference to Spanish if necessary
    if body_part_preference.lower() == "upper body":
        body_part_preference = "tren_superior"
    elif body_part_preference.lower() == "lower body":
        body_part_preference = "tren_inferior"
    elif body_part_preference.lower() == "back":
        body_part_preference = "espalda"
    
    # Filter exercises based on user preferences
    selected_type_exercises = gym_data[gym_data[body_part_preference] == 1].sample(5)

    # Get remaining types
    remaining_types = gym_data.columns.difference(['title', 'type', 'bodypart', 'equipment', body_part_preference])

    # Select one exercise from each of the remaining types if exercises are available
    remaining_exercises = pd.concat([gym_data[gym_data[column] == 1].sample(1) if gym_data[gym_data[column] == 1].shape[0] > 0 else pd.DataFrame() for column in remaining_types])

    # Combine selected exercises and remaining exercises
    all_selected_exercises = pd.concat([selected_type_exercises, remaining_exercises])

    return all_selected_exercises


def select_exercises_all(gym_data):
    # Translate column names to Spanish if necessary
    gym_data.columns = gym_data.columns.str.lower().str.replace(' ', '_')
    gym_data = gym_data.rename(columns={'upper_body': 'tren_superior', 'lower_body': 'tren_inferior', 'back': 'espalda'})

    # Select two exercises from each type
    selected_exercises = pd.concat([gym_data[gym_data['tren_superior'] == 1].sample(2),
                                    gym_data[gym_data['tren_inferior'] == 1].sample(2),
                                    gym_data[gym_data['core'] == 1].sample(2),
                                    gym_data[gym_data['espalda'] == 1].sample(2)])
    
    return selected_exercises


def get_table_download_link(df, file_name):
    """Generate a download link for a DataFrame in Excel format.

    Args:
        df (DataFrame): The DataFrame to download in Excel.
        file_name (str): The name of the Excel file.

    Returns:
        str: The HTML code for the download link.
    """
    # Convert DataFrame to Excel file
    df.to_excel(file_name, index=False)

    # Read Excel file as bytes
    with open(file_name, 'rb') as excel_file:
        excel_bytes = excel_file.read()

    # Encode bytes to base64
    b64 = base64.b64encode(excel_bytes).decode()

    # Generate download link
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download table as Excel</a>'

    return href

# Function to calculate BMI
def calculate_bmi(weight_kg, height_m):
    if height_m == 0:
        return float('inf')  # Return infinity if height is zero
    else:
        return weight_kg / (height_m ** 2)

# Function to calculate BMR
def calculate_bmr(age, weight_kg, height_m, gender_F, gender_M):
    if gender_F == 1:  # Female
        return 655 + (9.6 * weight_kg) + (1.8 * height_m * 100) - (4.7 * age)
    elif gender_M == 1:  # Male
        return 66 + (13.7 * weight_kg) + (5 * height_m * 100) - (6.8 * age)
    else:
        raise ValueError("Invalid gender values")

# Function to calculate daily calories
def calculate_daily_calories(user_inputs, initial_weight, desired_weight, time_interval, model):
    # Extract user inputs
    age = user_inputs['age']
    weight_kg = user_inputs['weight']
    height_m = user_inputs['height']
    gender_F = user_inputs['gender_F']
    gender_M = user_inputs['gender_M']
    activity_level = user_inputs['activity_level']

    # Calculate BMI and BMR using the provided formulas
    if height_m == 0:
        # Handle case when height is zero
        BMI = 0
        BMR = 0  # Assign a default value for BMR
    else:
        BMI = calculate_bmi(weight_kg, height_m)
        BMR = calculate_bmr(age, weight_kg, height_m, gender_F, gender_M)

    # Calculate total weight change
    weight_change = desired_weight - initial_weight
    calories_per_kg = 7700

    # Estimated calories per kg change (adjust as needed)
    daily_weight_change = weight_change / time_interval

    # Calculate caloric difference based on weight change goal
    caloric_difference = daily_weight_change * calories_per_kg

    # Create input array for model prediction
    input_array = np.array([[age, weight_kg, height_m, BMI, BMR, activity_level, gender_F, gender_M]])

    # Use the trained model to predict daily maintenance calories
    model_predicted_calories = model.predict(input_array)

    # Sum the model predicted calories and caloric difference
    daily_cal = model_predicted_calories + caloric_difference

    return daily_cal

def bmi_section(model):
    st.title("BMI Calculator")

    # Get user's weight
    weight = st.number_input("Enter your weight (kg)", min_value=0.0)  
    
    # Get user's height
    height = st.number_input("Enter your height (meters)", min_value=0.0)
    
    # Get user's age
    age = st.number_input("Enter your age", min_value=0, step=1)  # Age as an integer without decimals
    
    # Get user's gender (0 for No, 1 for Yes)
    gender = st.number_input("Are you a woman? (1 for Yes, 0 for No): ", min_value=0, max_value=1, step=1)  
    
    # Get user's activity level
    activity_level = st.number_input("""
    Enter your activity level between 0 and 100:
    0   - 0 minutes of exercise per week
    10  - 1 hour of exercise per week
    20  - 2 hours of exercise per week
    30  - 3 hours of exercise per week
    40  - 4 hours of exercise per week
    50  - 5 hours of exercise per week
    60  - 6 hours of exercise per week
    70  - 7 hours of exercise per week
    80  - 8 hours of exercise per week
    90  - 9 hours of exercise per week
    100 - 10+ hours of exercise per week
    """, min_value=0, max_value=100, step=10)

    initial_weight = weight 
    desired_weight = st.number_input("Enter your desired weight in kg: ", min_value=0.0)
    time_interval = st.number_input("Enter the time interval in days to reach your desired weight: ", min_value=0.1)  # Change minimum to 0.1
    
    # Check if time interval is greater than zero
    if time_interval == 0:
        st.error("The time interval cannot be zero. Please enter a value greater than zero.")
        return
    
    # Calculate BMI
    if weight > 0 and height > 0:
        bmi = calculate_bmi(weight, height)
        st.write(f"Your BMI is: {bmi:.2f}")
    
    # Calculate daily calories
    user_inputs = {
        'age': age,
        'weight': weight,
        'height': height,
        'gender_F': gender,
        'activity_level': activity_level
    }
    user_inputs['gender_M'] = 1 if user_inputs['gender_F'] == 0 else 0

    # Calculate daily calories using the previously defined function
    daily_calories = calculate_daily_calories(user_inputs, initial_weight, desired_weight, time_interval, model)

    st.write(f"Estimated daily calories: {daily_calories[0]}")
    
    # Show image below the content
    image_url = 'https://drcormillot.com.ar/wp-content/uploads/2022/02/imc_foto1.jpg'
    st.image(image_url, use_column_width=True)
    
    st.write(
    """
    While the Body Mass Index (BMI) is commonly used as an indicator of body fatness and health status, it has limitations. BMI does not always provide an accurate assessment of an individual's health due to several factors such as muscle mass, bone density, and body composition not being accounted for. As a result, individuals with high muscle mass may be categorized as overweight or obese, even if they have a low body fat percentage and are metabolically healthy. Conversely, individuals with a normal BMI may still have excess body fat and be at risk for health problems.
    
    Therefore, it's essential to recognize that BMI is just one tool among many for evaluating health and should be interpreted cautiously. It's always recommended to consult with healthcare professionals or experts in the field for a comprehensive assessment of one's health status, taking into account various factors beyond BMI.
    """
)

# Function to show the home section
def show_home():
    # Image URL from Google
    url_google_image = 'https://www.farmatopventas.es/images/easyblog_articles/62/nutrideporfarmatopventas.jpg'

    # Centered image
    st.image(url_google_image, use_column_width=True)
    st.title("Welcome to the Training and Nutrition Application")
    st.write(
    """
    Welcome to the Training and Nutrition Application! This app is designed to help you manage your health, fitness, and nutrition goals effectively. Let me guide you through its main features:

    1. **Home:** Start here to get an overview of the application and its purpose. You'll find essential information about maintaining a healthy lifestyle.

    2. **BMI Calculator:** Calculate your Body Mass Index (BMI) to assess your body composition and understand if you're underweight, normal weight, overweight, or obese.

    3. **Gym Section:** Explore various options to customize your gym experience. You can create personalized training plans or learn about different training types to achieve your fitness goals.

    4. **Recipes and Ingredients:** Discover healthy recipes and nutritional information for a wide range of foods. You can also study the nutrient distribution in your favorite recipes or foods.

    Feel free to navigate through the different sections and make the most out of the tools available. Remember, your health is in your hands!
    """
)

    st.header("Global Situation")
    st.write(
    """
    The global health landscape is undergoing a profound transformation, underscored by the World Health Organization's (WHO) sobering observations. An escalating tide of obesity, fueled by inadequate dietary habits and sedentary lifestyles, has emerged as a pressing public health concern.

    Recent epidemiological data paints a disquieting picture: more than 30% of the world's population grapples with overweight or obesity, ushering in a surge of chronic non-communicable diseases (NCDs). Type 2 diabetes, cardiovascular ailments, and certain malignancies loom large as formidable adversaries.

    The ramifications of this burgeoning health crisis are manifold, extending far beyond individual well-being to encompass societal and economic dimensions. To vividly illustrate this escalating predicament, let us embark on an exploration through three compelling visual narratives:

    1. **Global Obesity Trend (Average):** This graph encapsulates the relentless ascent of obesity prevalence rates across diverse populations, providing poignant insights into the overarching trajectory of this global phenomenon.

    2. **Obesity by Gender:** Delve into the intricate interplay of gender dynamics and obesity prevalence, as the graph delineates the disproportionate burden borne by females in the face of escalating adiposity levels.

    3. **Global Obesity Heatmap:** Embark on a journey across the globe with this comprehensive heatmap, meticulously crafted to showcase the geographical distribution of obesity prevalence. Through a vibrant spectrum of colors, this heatmap unveils the spatial heterogeneity in obesity rates, underscoring the multifaceted nature of this pervasive health challenge.
    """
)
    # Incorporate the global obesity trend graph
    st.header("Global Obesity Trend (Average)")
    st.components.v1.html(open("obesity_trend.html", "r").read(), height=600)
    
        # Incorporate the global obesity heatmap graph
    st.header("Obesity by Gender")
    st.components.v1.html(open("gender_obesity_comparison.html", "r").read(), height=600, width=1200) 
    
    # Incorporate the global obesity heatmap graph
    st.header("Global Obesity Heatmap")
    st.components.v1.html(open("obesity_heatmap.html", "r").read(), height=1000, scrolling=True, width=1200)

def select_training_type():
    st.write("Select the training type:")
    training_options = ["Power Strength", "Maximum Strength", "Strength Endurance","General Information"]
    training_type = st.radio("", training_options)
    return training_type

# Función para mostrar la información correspondiente al tipo de entrenamiento seleccionado
def display_information(strength_type, table):
    if strength_type.lower() == "power strength":
        st.write("Power Strength: Power, or explosive strength, encompasses the rapid generation of force, pivotal in sports and activities requiring agility and swift motions. Training for power mandates executing movements with maximal velocity, emphasizing explosiveness and speed. Should velocity diminish during exercises, adjusting the weight becomes imperative to sustain the intended training stimulus. It plays a crucial role in sports and activities requiring speed, agility, and quick movements.")
        st.write("Training for power involves lifting weights explosively at high velocities. This type of training stimulates fast-twitch muscle fibers and enhances neuromuscular coordination.")
        st.write("Key exercises for power development include plyometrics, Olympic weightlifting, and ballistic movements.")
        st.write("Information about Power Strength:")
        st.table(table.iloc[0:1])
        st.write("Sports where power strength is crucial include:")
        st.write("- Olympic Weightlifting")
        st.write("- Sprinting")
        st.write("- Basketball")
        st.write("- Volleyball")
    elif strength_type.lower() == "maximum strength":
        st.write("Maximum Strength: Maximum strength represents the peak force a muscle or muscle group can produce in a single maximal effort. It is essential for tasks requiring lifting heavy loads or overcoming resistance.")
        st.write("Maximal strength training involves lifting heavy weights at intensities close to 1-repetition maximum (1RM). This type of training promotes muscle hypertrophy and enhances force production.")
        st.write("Common exercises for maximal strength development include squats, deadlifts, bench presses, and other compound movements.")
        st.write("Information about Maximum Strength:")
        st.table(table.iloc[1:2])
        st.write("Sports where maximum strength is crucial include:")
        st.write("- Powerlifting")
        st.write("- Strongman competitions")
        st.write("- Rugby")
        st.write("- Wrestling")
    elif strength_type.lower() == "strength endurance":
        st.write("Strength Endurance: Strength endurance refers to the ability to sustain muscular contractions over an extended period. It is essential for activities requiring prolonged exertion, such as endurance sports or repetitive tasks.")
        st.write("Training for strength endurance involves using moderate weights for high repetitions or performing circuits with minimal rest. This type of training enhances muscular endurance and delays fatigue.")
        st.write("Exercises like bodyweight circuits, high-repetition resistance exercises, and endurance-based activities are effective for developing strength endurance.")
        st.write("Information about Strength Endurance:")
        st.table(table.iloc[2:])
        st.write("Sports where strength endurance is crucial include:")
        st.write("- CrossFit")
        st.write("- Marathon running")
        st.write("- Swimming")
        st.write("- Rowing")
    elif strength_type.lower() in ["general information", "general"]:
        st.write("General Strength Information: Strength training encompasses various aspects of physical strength, including power, maximum strength, and strength endurance. It is an integral part of overall fitness and performance enhancement.")
        st.write("Optimal strength development involves integrating all components of strength into a structured training program. A common approach is to cycle through different phases, focusing on power, maximum strength, and strength endurance sequentially.")
        st.write("Each type of strength training offers unique benefits and contributes to overall physical performance. While power training emphasizes speed and explosiveness, maximum strength training enhances force production, and strength endurance training improves muscular endurance.")
        st.write("Strength Training Table:")
        st.table(table)
    else:
        st.write("Sorry, the option is not recognized. Please try again.")
        
def set_background(image_file):
    # Leer el archivo de imagen como binario y convertirlo a una cadena codificada en base64
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    # Aplicar el estilo CSS para establecer la imagen de fondo
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded_string}');
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )    

def main():
    
    set_background('imagen_fondo9.png')
    # Load and preprocess the data once at the beginning
    model = load_and_preprocess_data()
    recipes_data = load_recipes_data()  # Load recipe data
    alimentos_clean = load_food_data()  # Load food data
    
    # Show image above the navigation menu
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2964/2964774.png", width=100, use_column_width=True)

    st.sidebar.title("Navigation Menu")
    option = st.sidebar.radio(
        "Select an option:", 
        ("Home", "BMI and Calories calculator", "Fitness seccion", "Recipes and Ingredients seccion")
    )

    if option == "Home":
        show_home()
        
    elif option ==  "BMI and Calories calculator":
        bmi_section(model)  # Pass 'model' as an argument to the bmi_section() function
        
    elif option == "Fitness seccion":
        st.title("Gym")
        gym_option = st.sidebar.radio(
            "Select an option:",
            ("Training Plan", "Training Type")
        )

        if gym_option == "Training Plan":
            st.title("Gym Training Plan")
            # Add image
            st.image("https://media.foodspring.com/magazine/public/uploads/2020/10/trainingsplan.jpg", use_column_width=True)
            generate_training_plan(gym_data)
        elif gym_option == "Training Type":
            st.title("Training Type in the Gym")
            training_type = select_training_type()  # Capture the selected training type
            if training_type:
                # Show information corresponding to the selected training type
                display_information(training_type, create_strength_table())
                # Add the photo at the end of the training type page
                st.image("https://www.feda.net/wp-content/uploads/2018/08/circuit-training.jpeg", use_column_width=True)
            
    elif option == "Recipes and Ingredients seccion":
        st.title("Recipes and Ingredients")
        recipes_ingredients_option = st.sidebar.radio(
            "Select an option:",
            ("Recipe Ideas", "Recipe Study", "Food Study")
        )

        if recipes_ingredients_option == "Recipe Ideas":
            generate_recipe_ideas_section(recipes_data)  # Pass 'recipes_data' as an argument

        elif recipes_ingredients_option == "Recipe Study":
            st.title("Recipe Study")
            # Get the recipe name from the user
            recipe_name = st.text_input("Enter the recipe name:")
            
            st.image("https://img.chemie.de/Portal/News/6526753d81977_pEetMoyxm.png?tr=w-1200,h-600,cm-extract,x-0,y-215:n-news_teaser", use_column_width=True)
            
            # Check if a recipe name is entered
            if recipe_name:
                # Search for recipe information by its name
                recipe_data = recipes_data[recipes_data['name'] == recipe_name]
                
                # Check if the recipe is found
                if not recipe_data.empty:
                    # Show the donut chart
                    plot_nutrient_distribution(recipe_data)
                    
                    # Show calorie information
                    display_calories_info(recipe_data)
                    
                else:
                    st.write("The specified recipe was not found in the list of available recipes.")
        
        elif recipes_ingredients_option == "Food Study":
            st.title("Food Study")
            # Get the food name from the user
            food_name = st.text_input("Enter the food name:")
            
            st.image("https://img.chemie.de/Portal/News/6526753d81977_pEetMoyxm.png?tr=w-1200,h-600,cm-extract,x-0,y-215:n-news_teaser", use_column_width=True)
            
            # Check if a food name is entered
            if food_name:
                # Search for food information by its name
                food_data = alimentos_clean[alimentos_clean['name'].str.contains(food_name, case=False)]
                
                # Check if the food is found
                if not food_data.empty:
                    # Show the donut chart for the food
                    plot_nutrients_distribution(food_data, food_name)
                    
                else:
                    st.write(f"No foods containing '{food_name}' were found.")

# Call the main function to start the application
if __name__ == "__main__":
    main()