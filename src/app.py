import streamlit as st
import pandas as pd
import sys
import os
import ast
import traceback
import re

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
    """Load and cache the agent"""
    config = ags.AgentConfig(max_steps=6, verbose=True)
    agent_instance = ags.ReActAgent(lm.LLM, kb.TOOLS, config)
    return agent_instance

@st.cache_data
def load_recipe_data():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")

    cleaned_path = os.path.join(DATA_PATH, "recipes_cleaned.csv")
    raw_path = os.path.join(DATA_PATH, "recipes.csv")

    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        df = df.rename(columns={
            "recipe_name": "Name",
            "total_time": "Total Time",
            "ingredients": "Ingredients",
            "directions": "Directions",
            "url": "URL"
        })
    else:
        st.error("No recipe data found in data/.")
        return pd.DataFrame()
    
    df['Name'] = df['Name'].fillna("Unnamed Recipe")
    df['Total Time'] = df['Total Time'].fillna("Unknown")
    df['Ingredients'] = df['Ingredients'].fillna("[]")
    return df

def parse_ingredients_for_display(ing_data):
    """Handles different ingredient formats (String vs List of Dicts)."""
    if isinstance(ing_data, str):
        if ing_data.strip().startswith("[{"):
            try:
                ing_list = ast.literal_eval(ing_data)
                return [f"{i.get('quantity', '')} {i.get('unit', '')} {i.get('name', '')}" for i in ing_list]
            except:
                pass
        return [ing_data]
    return ["No ingredients listed."]

# ==========================================
# INITIALIZE
# ==========================================
try:
    agent = load_agent()
    recipes_data = load_recipe_data()
except Exception as e:
    st.error(f"Failed to load agent or data: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ==========================================
# UI LAYOUT
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
# DISPLAY CHAT HISTORY
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# HANDLE SEARCH BUTTON CLICK
# ==========================================
if search_btn:
    if not ingredients_input:
        st.warning("Please enter some ingredients to start!")
    else:
        # Build query
        if time_input > 0:
            user_query = f"What can I make using {ingredients_input}? The recipe should take around {time_input} minutes or less."
        else:
            user_query = f"What can I make using {ingredients_input}?"
        
        # Add to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking and searching..."):
                try:
                    result = agent.run(user_query)
                    #st.write("DEBUG RESULT:", result)
                    
                    final_answer = result.get("final_answer")
                    #st.write("DEBUG final_answer:", final_answer)

                    if not final_answer or not isinstance(final_answer, str):
                        msg = "Sorry — I couldn't find a specific recipe based on that request."
                        st.error("The agent did not return a valid recipe name.")
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        st.write(msg)
                    else:
                        # Display answer
                        st.write(final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})

                        st.markdown("---")
                        found_match = False

                        # Try to extract recipe names from the answer
                        # Handle both "Recipe Name" and "You can make Recipe Name by..."
                        potential_names = []
                        
                        # If answer is a sentence, try to extract recipe name
                        if "make" in final_answer.lower() or "by" in final_answer.lower():
                            # Extract the recipe name from sentences like "You can make X by..."
                            patterns = [
                                r'make\s+([A-Z][^.!?]*?)(?:\s+by|\s+with|\.|$)',
                                r'recipe:?\s*([A-Z][^.!?]*?)(?:\.|$)',
                                r'called\s+([A-Z][^.!?]*?)(?:\.|$)',
                            ]
                            for pattern in patterns:
                                match = re.search(pattern, final_answer)
                                if match:
                                    potential_names.append(match.group(1).strip())
                        
                        # Also try splitting by comma or period
                        if not potential_names:
                            potential_names = [name.strip() for name in re.split(r'[,.]', final_answer)]
                        
                        # Clean and deduplicate
                        potential_names = [name for name in potential_names if name and len(name) > 3]
                        
                        for potential_name in potential_names:
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
                    st.code(traceback.format_exc())