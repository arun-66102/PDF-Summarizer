# department_corpus.py

DEPARTMENT_CORPUS = {
    "CSE": {
        "department_name": "Computer Science and Engineering",
        "description": (
            "Computer Science and Engineering focuses on software development, programming, algorithms, "
            "data structures, databases, artificial intelligence, machine learning, cloud computing, "
            "cybersecurity, web development, mobile apps, data science, computer networks, "
            "operating systems, software engineering, and computational theory."
        ),
        "keywords": [
            "programming", "algorithm", "data structure", "machine learning",
            "artificial intelligence", "database", "sql", "python", "java",
            "cloud computing", "cyber security", "software development", "web development",
            "mobile apps", "data science", "computer networks", "operating systems",
            "software engineering", "computational theory", "deep learning", "neural networks",
            "computer vision", "natural language processing", "big data", "devops", "kubernetes",
            "docker", "react", "angular", "nodejs", "tensorflow", "pytorch", "keras"
        ]
    },

    "EEE": {
        "department_name": "Electrical and Electronics Engineering",
        "description": (
            "Electrical and Electronics Engineering deals with power systems, electrical machines, "
            "control systems, power electronics, circuits, embedded systems, signal processing, "
            "renewable energy, smart grids, telecommunications, and electronic design automation."
        ),
        "keywords": [
            "power systems", "control systems", "electrical machines",
            "power electronics", "circuit analysis", "embedded systems", "signal processing",
            "renewable energy", "smart grid", "telecommunications", "electronic design",
            "transformer", "generator", "motor", "voltage", "current", "resistance",
            "capacitance", "inductance", "pcb", "microcontroller", "arduino", "raspberry pi",
            "matlab", "simulink", "power transmission", "distribution system"
        ]
    },

    "MECH": {
        "department_name": "Mechanical Engineering",
        "description": (
            "Mechanical Engineering involves thermodynamics, fluid mechanics, heat transfer, "
            "machine design, manufacturing processes, CAD/CAM, automobiles, materials science, "
            "finite element analysis, robotics, mechatronics, and HVAC systems."
        ),
        "keywords": [
            "thermodynamics", "fluid mechanics", "heat transfer", "manufacturing",
            "cad", "cam", "cnc", "machine design", "automobile", "materials science",
            "finite element analysis", "robotics", "mechatronics", "hvac", "refrigeration",
            "air conditioning", "pumps", "turbines", "engines", "gears", "bearings",
            "welding", "casting", "forging", "metrology", "quality control", "autocad",
            "solidworks", "ansys", "catia", "pro engineer", "solid edge"
        ]
    },

    "CIVIL": {
        "department_name": "Civil Engineering",
        "description": (
            "Civil Engineering focuses on structural engineering, construction planning, surveying, "
            "geotechnical engineering, concrete technology, environmental engineering, transportation "
            "engineering, water resources, urban planning, and infrastructure development."
        ),
        "keywords": [
            "structural engineering", "construction planning", "surveying", "geotechnical engineering",
            "concrete technology", "environmental engineering", "transportation engineering",
            "water resources", "urban planning", "infrastructure", "foundation", "beam",
            "column", "slab", "retaining wall", "bridge", "road", "highway", "railway",
            "airport", "sewage", "water treatment", "irrigation", "dam", "tunnel",
            "building materials", "steel", "cement", "aggregate", "soil mechanics", "estimation",
            "costing", "project management", "autocad civil", "staad pro", "etabs", "sap2000"
        ]
    }
}


def build_department_documents():
    documents, metadatas, ids = [], [], []

    for code, dept in DEPARTMENT_CORPUS.items():
        text = f"""
        Department: {dept['department_name']}
        Description: {dept['description']}
        Keywords: {', '.join(dept['keywords'])}
        """

        documents.append(text.strip())
        metadatas.append({
            "department_code": code,
            "department_name": dept["department_name"]
        })
        ids.append(code)

    return documents, metadatas, ids