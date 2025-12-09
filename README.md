# CS 4100 Course Project: Personal Health Wellness Agent

### Team Members: Christopher Le and Sam Castelein

Course: CS 4100 Artificial Intelligence (Fall 2025)

Insallation instructions will be found in INSTALL.md.

## Abstract
The Personal Health & Wellness Agent is an AI-powered system designed to create personalized weekly meal and workout plans. We can counter the unhelpful fitness advice found online by using an AI agent. By using the 4-step pipeline we were asked to use: parsing user queries, retrieving factual data from local databases, prompting with context, and generating a cohesive plan using a generative transformer model.

## Overview
The recent boom in health and fitness has left many beginners confused by contradictory online advice. While professional coaching is effective, it is often prohibitively expensive. Beginners need a way to get structured, safe, and personalized guidance without sifting through endless articles or paying for a trainer.

## Why It Matters
This project democratizes access to personalized health planning. Instead of a one-size-fits-all PDF, users get a plan that respects their nut allergy, their "intermediate" fitness level, and their specific goal to "build muscle." 

## Approach
Our agent operates on a sequential 4-step pipeline that we were introduced in class. After cleaning our initial csv file and creating a corpus we did the following:

1. Intent Recognition: A parsing module extracts key entities (e.g., "vegetarian", "beginner") from the user's natural language query.
2. Intelligent Search (Retrieval): We filter structured CSV/JSON datasets to find recipes and exercises that strictly match the parsed constraints.
3. Contextual Prompting: We dynamically construct a prompt that includes the user's profile and the retrieved list of valid items.
4. Generation: We feed this context-rich prompt into a pre-trained LLM (Qwen2.5-0.5B-Instruct or distilgpt2) to synthesize the final weekly plan.

## Algorithm & Model
1. Search Algorithm: Boolean filtering using the pandas library. This acts as a deterministic guardrail.
2. Generative Model: We utilized the Qwen/Qwen2.5-0.5B-Instruct model from Hugging Face. This is a decoder-only transformer model optimized for instruction following.

## Experiments

### Dataset
We curated two primary datasets for this project:
1. recipes.csv: A collection of recipes with nutritional metadata (Calories, Vegetarian/Vegan tags, Gluten-Free tags). Sourced from Kaggle and manually cleaned.
2. exercises.json: A structured list of exercises categorized by difficulty level (Beginner, Intermediate, Advanced) and target muscle group.

### Implementation Details
Environment: The project was developed and executed in Google Colab using a T4 GPU runtime.
Libraries: transformers, torch, pandas, accelerate.
Hyperparameters: For generation, we used a temperature of 0.7 to balance creativity with coherence, and max_new_tokens=1024 to ensure the model had enough "space" to write a full 3-day plan.


## Results

### Main Results

The agent successfully generated coherent, constraint-compliant plans 100% of the time for standard queries.

Constraint Satisfaction: When tested with the query "I'm a vegetarian with a nut allergy," the agent correctly retrieved only recipes marked is_vegetarian=True and is_nut_free=True, never hallucinating a meat-based dish.

Level Appropriateness: When a user specified "beginner," the agent successfully filtered out "Advanced" moves like Barbell Squats, suggesting Bodyweight Squats instead.

### Example


## Discussion

### Comparison & Limitations

Our "Search-then-Generate" approach proved superior to a "Generate-only" baseline. A standard LLM asked to "make a meal plan" often invents recipes or ignores calorie limits. Our agent cannot violate the hard constraints because it is only allowed to choose from the pre-filtered list.

However, the current system has limitations:
- Vocabulary Gap: The parser relies on keyword matching. If a user says "I don't eat meat" instead of "vegetarian," the simple parser might miss it.
- Database Size: The quality of the output is strictly limited by the size of our database. If we have no "vegan keto" recipes, the agent cannot return a plan.


## Conclusion

We successfully built a functional AI agent that bridges the gap between structured data retrieval and natural language generation. By constraining a powerful LLM with verified data, we created a tool that provides safe, personalized, and actionable health advice.

## References
**Bibliography**

[1] Dataset Source: Kaggle.com Recipes Dataset. https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life

[2] Dataset Source: Kaggle.com Excercise Dataset. https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset/data

[3] Model: Qwen2.5-0.5B-Instruct. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

