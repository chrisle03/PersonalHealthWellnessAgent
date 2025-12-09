import streamlit as st
import pandas as pd
import sys
import os
import ast

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
    import agent_system as ags
    import knowledge_base as kb
    import language_model as lm
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Make sure your project structure has 'src/' containing agent_system.py, knowledge_base.py, etc.")
    st.stop()


st.set_page_config(
    page_title="Health & Wellness Agent",
    page_icon="🍳",
    layout="wide"
)

@st.cache_resource
def load_agent():
    config = ags.AgentConfig(max_steps=6, verbose=True)
    return ags.ReActAgent(lm.LLM, kb.TOOLS, config)

@st.cache_data
def load_recipe_data():
    if os.path.exists("recipes_cleaned.csv"):
        df = pd.read_csv("recipes_cleaned.csv")
    elif os.path.exists("recipes.csv"):
        df = pd.read_csv("recipes.csv")
        df = df.rename(columns={'recipe_name': 'Name', 'total_time': 'Total Time', 'ingredients': 'Ingredients'})
    else:
        st.error("No recipe data found. Please run 'data_processing.py' first.")
        return pd.DataFrame()
    
    df['Name'] = df['Name'].fillna("Unnamed Recipe")
    df['Total Time'] = df['Total Time'].fillna("Unknown")
    df['Ingredients'] = df['Ingredients'].fillna("[]")
    return df

try:
    agent = load_agent()
    recipes_data = load_recipe_data()
except Exception as e:
    st.error(f"Failed to load agent or data: {e}")
    st.stop()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def parse_ingredients_for_display(ing_data):
    """
    Handles different ingredient formats (String vs List of Dicts).
    """
    if isinstance(ing_data, str):
        # Case A: String representation of a list of dicts "[{'name': 'salt'...}]"
        if ing_data.strip().startswith("[{"):
            try:
                ing_list = ast.literal_eval(ing_data)
                return [f"{i.get('quantity', '')} {i.get('unit', '')} {i.get('name', '')}" for i in ing_list]
            except:
                pass # Fallback to plain string
        
        # Case B: Plain string "2 eggs, 1 cup flour"
        return [ing_data]
        
    return ["No ingredients listed."]

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.title("🍳 AI Recipe & Wellness Assistant")
st.markdown("I can help you find recipes based on ingredients and cooking time!")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Search Parameters")
    ingredients_input = st.text_input("Ingredients (e.g., chicken, rice):")
    time_input = st.number_input("Max Cooking Time (minutes):", min_value=0, step=10, value=0)
    
    search_btn = st.button("Find Recipes", type="primary")
    
    st.markdown("---")
    st.markdown("**Agent Status:** Ready")
    st.caption(f"Loaded {len(recipes_data)} recipes.")

# ==========================================
# 5. APP LOGIC
# ==========================================

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if search_btn and ingredients_input:
    if time_input > 0:
        user_query = f"What can I make using {ingredients_input}? The recipe should take around {time_input} minutes or less."
    else:
        user_query = f"What can I make using {ingredients_input}?"
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Run Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching..."):
            try:
                result = agent.run(user_query)
                final_answer = result.get("final_answer", "I couldn't find a specific answer.")
                
                st.write(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

                st.markdown("---")
                found_match = False
                

                potential_names = final_answer.split(", ")
                
                for potential_name in potential_names:
                    # Clean the name slightly
                    clean_name = potential_name.strip(".").strip()
                    match = recipes_data[recipes_data["Name"].str.contains(clean_name, case=False, regex=False)]
                    
                    if not match.empty:
                        found_match = True
                        recipe = match.iloc[0]
                        
                        with st.expander(f"📖 View Recipe: {recipe['Name']}", expanded=True):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.metric("Total Time", f"{recipe['Total Time']} m")
                                if 'Calories' in recipe:
                                    st.metric("Calories", f"{recipe['Calories']}")
                                if recipe.get('URL'):
                                    st.link_button("Go to Website", recipe['URL'])
                                    
                            with col2:
                                st.subheader("Ingredients")
                                ing_list = parse_ingredients_for_display(recipe['Ingredients'])
                                for item in ing_list:
                                    st.markdown(f"- {item}")
                                    
                                if 'Directions' in recipe and pd.notna(recipe['Directions']):
                                    st.subheader("Directions")
                                    st.caption(str(recipe['Directions'])[:500] + "...")

                if not found_match:
                    st.caption("Detailed recipe cards could not be loaded automatically based on the agent's answer.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

elif search_btn and not ingredients_input:
    st.warning("Please enter some ingredients to start!")
