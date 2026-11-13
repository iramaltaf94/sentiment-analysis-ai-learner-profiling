# Sentiment Analysis & AI Profiling of Learners
**Live dashboard:** https://learner-analytics-iram.streamlit.app

A team project (3 members) done between April 2026 and July 2026, with the approval of Chinar Quantum AI.

## What this project is about

Learner evaluations often reduce a person to one score. We wanted to see what mentoring conversations show about a learner's confidence and hesitation that a single number cannot. We built a Python pipeline that combines two kinds of data: structured live observation scores, and unstructured session notes and transcripts. Text analysis and an LLM then turn both into readable learner profiles.

## Where and on whom

Chinar Quantum AI runs an introductory AI course called "Future AI" (FAI) for new learners, with one-on-one mentoring. We worked on real data from these sessions, covering 19 learners. All learner names in this repository are changed to Student 1 to Student 19. The results are real. Nothing is made up. The original data files are not shared.

## How the team worked

Structured data (live observation):
- Three team members observed learners live during the sessions, each on one area: vocal, linguistic and interaction flow. Each area has its own data table.
- The observations were scored on a scale from -2 (highly negative) to +2 (highly positive), with 0 as neutral. The three tables were then joined on the participant and the scores were converted into normalized weights.
- A rule-based classification then placed each learner into a profile type such as Star Performance, Hesitant Expert, Enthusiastic Learner, Knowledge Gap, Steady Contributor or Balanced Profile. Each learner also got an insight and a suggested learning track (for example Advanced, Technical, Foundational, Interactive or Remedial).
- A radar-style chart (neural profile) was made for each learner.

Unstructured data (text):
- Session transcripts and notes were analysed with NLTK VADER, and the Groq LLM was used to write profile cards.

360 degree profile:
- For every learner, the unified notebook puts all of this in one place: live observation data, structured insights and the chart, the VADER text analysis, and the AI profile.

Proposed for the future, but not built: continuous, quantum-inspired sentiment modelling. The quantum wavefunction panel in the unified notebook is a visualization only.

## My contribution

Structured data:
- I did the interaction flow observation, one of the three areas, and scored the learners on it. This is the flow data in the unified notebook.

Unstructured data and AI profiling:
1. Cleaned the mentors' Gemini-generated session notes. I removed unclear passages and fixed words only where the context was certain.
2. Built the learner CSV from these notes.
3. Text analysis with NLTK VADER for the 18 learners whose transcripts were available. For each learner I calculated the sentiment, a confidence score out of 10 (from keyword lists) and an estimate of cognitive load (from hesitation words), shown as bar charts. The code is a reusable function that works for every learner in the data.
4. Profile cards with the Groq LLM API. I wrote the prompt so that the model creates one profile card for each of the 19 learners. Each card shows education, what the learner said about the organization, goals, current status, strengths, and where support is needed. The rules are strict: only stated facts, mentor remarks are never given to the learner, and personal details are left out unless the learner said them.

How the text scores work:
- Sentiment: VADER gives a score from -1 to +1. Above 0.05 is positive, below -0.05 is negative, and in between is neutral.
- Confidence score: high-confidence words found divided by all words found, multiplied by 10. If no word is found, the score is 5.
- Cognitive load: hesitation words like uh, umm, actually, like, sorry and for example are counted. 0 to 2 is Low, 3 to 5 is Moderate, and more than 5 is High.

## Outcome

- 19 AI-written learner profile cards, together with the observation scores and the text analysis where available.
- It showed that hesitation does not always mean a knowledge gap. Some strong learners had low vocal confidence.
- It is designed to help mentors see who needs encouragement and who is ready for more.

## Files

- 01_Iram_Text_Analysis_NLTK_VADER.ipynb: my text analysis (18 learners)
- 01_Iram_Text_Analysis_NLTK_VADER_Preview.pdf: how my text analysis looks
- 02_Iram_AI_Profile_Cards_Groq.ipynb: my profile cards (19 learners)
- 02_Iram_AI_Profile_Cards_Groq_Preview.pdf: how my profile cards look
- 03_Team_Unified_Analytics_Platform.ipynb: the unified notebook of the whole team. It has the structured live observation data, the AI profiling summary, the classification, the text analysis and a 360 degree profile for each learner. My parts above are included in it.

## How to view

Open any notebook on GitHub and you can see the results that were saved in it. In the unified notebook, my profile cards already look different from the original design, because GitHub shows the colours and styling in a simpler way. The real design of my profile cards is in the PDF 02_Iram_AI_Profile_Cards_Groq_Preview.pdf.

The notebooks cannot be run here because they need the private data and a Groq API key, which are not shared.

## Tools used

Python, Pandas, NumPy, NLTK, VADER, Groq API, Matplotlib, Google Colab, Prompt Engineering
