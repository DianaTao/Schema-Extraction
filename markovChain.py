import random

# ----------------------------
# Sample dataset: emails
# ----------------------------
emails = [
    """Subject: Exciting Opportunities for Collaboration

Hi Jennifer,

I hope this message finds you well! I am reaching out to explore potential collaboration opportunities between Orion Creative Studios and your team at Galaxy Innovations. We're impressed with your latest project, 'Stellar Visions,' and believe our expertise in visual effects could bring significant value to your upcoming ventures.

Let's schedule a meeting to discuss how we can align our efforts and create something extraordinary together. Please let me know your availability next week, and we'll coordinate a time that suits both parties.

Looking forward to the possibilities!

Best regards,

David Chen
Chief Creative Officer
Orion Creative Studios
San Francisco, CA

P.S. Have you had a chance to try out the new Genesis Pro? We're getting remarkable feedback from the creative community and would love to hear your thoughts!""",
    
    """Subject: Partnership Proposal with Nexus Enterprises

Dear Michael,

I hope this email finds you in great spirits! I am reaching out on behalf of Aurora Technologies to propose a partnership with Nexus Enterprises. Your innovative approach, especially demonstrated in the 'Infinity Cycle' series, aligns perfectly with our vision for technological advancement.

We are particularly interested in discussing how our latest product, the Quantum Plus, could enhance your projects and drive mutual growth in our respective markets.

Could we arrange a meeting to delve deeper into these opportunities? Let me know a suitable time for you next week.

Warm regards,

Emma Torres
Partnership Manager
Aurora Technologies""",
    
    """Subject: Opportunity for Collaboration

Dear Alex,

My name is Lisa Kim, and I am the Director of Partnerships at Neptune Studios. We are incredibly impressed by the innovative work your team at Stellar Dynamics has been doing, particularly with your latest documentary series, 'Cosmic Journeys.'

We believe there's a fantastic opportunity for collaboration between our two companies, especially considering our recent launch of the Orion XR headset, which could offer an immersive viewing experience for your content.

Would you be available for a meeting next week to explore how we could potentially work together on future projects? Please share your availability, and I'll do my best to accommodate.

Looking forward to the possibility of a rewarding partnership.

Kind regards,

Lisa Kim
Director of Partnerships
Neptune Studios""",
    
    """Subject: Exploring Synergy between SynthWave Audio and Harmonic Ventures

Hi James,

I'm Sofia Martinez, Business Development Lead at SynthWave Audio. I've been following the exciting work your team at Harmonic Ventures has been delivering, notably your recent audio engineering masterpiece, 'Echoes of the Future.'

We believe there's immense potential in joining forces, leveraging our newly launched product, the Sonata Crystal speaker series, acclaimed for its unparalleled acoustics. Together, we could redefine listening experiences and tap into new markets.

Could we arrange a time to discuss this potential collaboration? Kindly inform me of your availability in the coming days.

Eager to hear your thoughts.

Best regards,

Sofia Martinez
Business Development Lead
SynthWave Audio"""
]

# ----------------------------
# Step 1. Extract Attributes from Emails
# ----------------------------
# For demonstration, we hard-code the extraction results.
# In a real-world system, you might use NLP/regex to extract these.
extracted_relationships = [
    {
        "sender": "David Chen",
        "title": "Chief Creative Officer",
        "company": "Orion Creative Studios",
        "recipient": "Jennifer",       # from greeting: "Hi Jennifer,"
        "subject": "Exciting Opportunities for Collaboration"
    },
    {
        "sender": "Emma Torres",
        "title": "Partnership Manager",
        "company": "Aurora Technologies",
        "recipient": "Michael",       # from "Dear Michael,"
        "subject": "Partnership Proposal with Nexus Enterprises"
    },
    {
        "sender": "Lisa Kim",
        "title": "Director of Partnerships",
        "company": "Neptune Studios",
        "recipient": "Alex",          # from "Dear Alex,"
        "subject": "Opportunity for Collaboration"
    },
    {
        "sender": "Sofia Martinez",
        "title": "Business Development Lead",
        "company": "SynthWave Audio",
        "recipient": "James",         # from "Hi James,"
        "subject": "Exploring Synergy between SynthWave Audio and Harmonic Ventures"
    }
]

# ----------------------------
# Step 2. Build a Probabilistic Graph (Adjacency List)
# ----------------------------
# We use a dictionary where keys are nodes (attributes) and values are dicts mapping related nodes to probabilities.
# In this example, we define direct relationships from the extracted relationships.
graph = {}

def add_edge(source, target, probability):
    """Helper function to add an edge from source to target with the given probability."""
    if source not in graph:
        graph[source] = {}
    # If an edge already exists, we could average or sum probabilities; here we override for simplicity.
    graph[source][target] = probability

# For each email, add edges for sender->title and title->company.
for relation in extracted_relationships:
    sender = relation["sender"]
    title = relation["title"]
    company = relation["company"]
    # Assume a high probability for direct observation:
    add_edge(sender, title, 0.9)
    add_edge(title, company, 0.8)
    
    # Optionally, add an edge from sender to recipient based on greeting (lower probability)
    recipient = relation["recipient"]
    add_edge(sender, recipient, 0.6)
    
    # Also, we can connect subject to sender as a loose association:
    subject = relation["subject"]
    add_edge(subject, sender, 0.5)

# For demonstration, let's also add some cross-attribute edges between titles and inferred secondary roles.
# For instance, a Director of Partnerships might imply a relationship to "Business Development" roles.
add_edge("Director of Partnerships", "Business Development Lead", 0.4)
add_edge("Chief Creative Officer", "Director of Partnerships", 0.3)

# ----------------------------
# Step 3. Probabilistic Sampling and Inference Functions
# ----------------------------
def sample_next_node(current_node):
    """
    Given a current node, probabilistically select the next node based on outgoing edges.
    """
    if current_node not in graph or not graph[current_node]:
        return None
    next_nodes = list(graph[current_node].keys())
    probabilities = list(graph[current_node].values())
    # Normalize probabilities in case they don't sum to 1
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    return random.choices(next_nodes, weights=probabilities, k=1)[0]

def infer_probability(start_node, target_node):
    """
    If there is a direct edge, return its probability.
    Otherwise, try a two-step inference by multiplying probabilities along a two-edge path.
    Returns the maximum inferred probability found.
    """
    # Direct edge
    if start_node in graph and target_node in graph[start_node]:
        return graph[start_node][target_node]
    
    # Check two-step paths
    inferred_probs = []
    if start_node in graph:
        for intermediate, p1 in graph[start_node].items():
            if intermediate in graph and target_node in graph[intermediate]:
                p2 = graph[intermediate][target_node]
                inferred_probs.append(p1 * p2)
    if inferred_probs:
        return max(inferred_probs)
    return 0.0

def generate_statement(entity):
    """
    Generate a natural language statement for an entity by sampling its relationships.
    For example, for a sender, output their title, inferred company, and even an inferred relationship.
    """
    # First sample a direct relationship from the entity.
    direct_relation = sample_next_node(entity)
    if not direct_relation:
        return f"No relationship found for {entity}."
    
    statement = f"{entity} is associated with {direct_relation}"
    
    # Attempt to infer a secondary relationship.
    inferred = []
    if entity in graph:
        for intermediate in graph[entity]:
            for secondary in graph.get(intermediate, {}):
                # Compute inferred probability as the product of edges.
                inferred_prob = graph[entity][intermediate] * graph[intermediate][secondary]
                # Plausibility filtering: only consider if above a threshold (e.g., 0.2)
                if inferred_prob >= 0.2 and secondary != direct_relation:
                    inferred.append((secondary, round(inferred_prob, 2)))
    
    if inferred:
        inferred_str = "; ".join([f"{sec} ({prob})" for sec, prob in inferred])
        statement += f". Inferred relations: {inferred_str}"
    else:
        statement += "."
    
    return statement

# ----------------------------
# Step 4. Demonstrate the Model with the Email Dataset
# ----------------------------
entities_to_test = [
    "David Chen",
    "Emma Torres",
    "Lisa Kim",
    "Sofia Martinez",
    "Exciting Opportunities for Collaboration"  # testing subject as a starting node
]

print("Probabilistic Relationship Statements from Email Dataset:\n")
for entity in entities_to_test:
    print(generate_statement(entity))

# ----------------------------
# Optional: Show full graph structure (for debugging/demonstration)
# ----------------------------
print("\nGraph Structure:")
for node, edges in graph.items():
    print(f"{node} -> {edges}")
