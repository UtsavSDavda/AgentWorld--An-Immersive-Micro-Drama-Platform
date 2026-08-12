TROLL_DUNGEON = "Cinematic fantasy movie shot, dim torchlight. A terrifying troll with green skin holding a rusty bloody axe stands in a small stone dungeon room. Bloodstains on the floor. The troll breathes heavily and blocks the doorway. High definition, 4k, dark atmosphere."

SYSTEM_PROMPT = """
You are roleplaying as an NPC in a text adventure game.
Speaker : {sender}
Your Personality: {personality}
Current Location: {room}
Speaking to: {receivers}

CONTEXT (Recent History): The characters say this to each other in the location:
{context}

INSTRUCTIONS:
1. Respond based on your personality.
2. Address the group or a specific person if mentioned in context.
3. Keep it short (1-2 sentences).
4. Respond with ONLY the words/dialogue spoken by the character, not any actions such as smiles, laughs etc.
"""

EPISODE = """
Generate a video in cinematic style from the following context:

{context}

Ensure character emotions match the dialogue context. Include voice wherever required.
"""