import torch

# Example Sentences (30 samples: 15 Sports, 15 Politics)
sentences = [
    # Sports (5 Negative, 5 Neutral, 5 Positive)
    "The Lakers suffered a devastating defeat in the NBA playoffs.",
    "Cristiano Ronaldo's missed penalty cost his team the match.",
    "The Olympics faced multiple organizational issues this year.",
    "A major doping scandal shook the athletics world.",
    "The FIFA World Cup final was overshadowed by referee controversies.",
    "The team's performance was average, neither too great nor too bad.",
    "The new rule changes brought mixed reactions from players and coaches.",
    "The match ended in a goalless draw after extra time.",
    "The new stadium is under construction and expected to open next year.",
    "The player’s transfer to another club is still under negotiation.",
    "Lewis Hamilton won the Formula 1 Grand Prix.",
    "Max Verstappen dominated the F1 season with multiple wins.",
    "Ferrari showed impressive speed improvements this season.",
    "The Monaco Grand Prix remains one of the most prestigious races in F1.",
    "Mercedes successfully implemented a new aerodynamic package.",
    # Politics (5 Negative, 5 Neutral, 5 Positive)
    "The Supreme Court ruling sparked nationwide protests.",
    "A diplomatic scandal led to tensions between countries.",
    "The government’s response to the crisis was heavily criticized.",
    "Voter suppression concerns rise ahead of the national elections.",
    "Public dissatisfaction with leadership is at an all-time high.",
    "The President delivered a speech on economic reform.",
    "Congress passed a new climate change bill this week.",
    "The United Nations is hosting a summit on global security.",
    "A major political debate took place ahead of the elections.",
    "New policies on immigration were announced today.",
    "The country saw significant economic growth this quarter.",
    "A peace agreement was signed after months of negotiations.",
    "The government launched a new initiative to improve education.",
    "International cooperation led to a breakthrough in trade relations.",
    "The infrastructure project was completed ahead of schedule."
]

# Labels for Task A (Sentence Classification)
labels_taskA = torch.tensor([0]*15 + [1]*15)  # 0 = Sports, 1 = Politics

# Labels for Task B (Sentiment Analysis)
labels_taskB = torch.tensor([
    0, 0, 0, 0, 0,  # Sports Negative
    1, 1, 1, 1, 1,  # Sports Neutral
    2, 2, 2, 2, 2,  # Sports Positive
    0, 0, 0, 0, 0,  # Politics Negative
    1, 1, 1, 1, 1,  # Politics Neutral
    2, 2, 2, 2, 2   # Politics Positive
])
