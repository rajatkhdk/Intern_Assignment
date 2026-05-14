from __future__ import annotations
from collections import defaultdict

# ── Skill Taxonomy ─────────────────────────────────────────────────────────────
# Each category maps to a set of representative keywords.
# Keywords are matched against the cleaned (lemmatised, lowercase) token list.

SKILL_TAXONOMY: dict[str, set[str]] = {
    "Programming Languages": {
        "python", "java", "javascript", "typescript", "c", "cpp", "csharp",
        "ruby", "go", "golang", "rust", "swift", "kotlin", "scala", "php",
        "bash", "shell", "r", "matlab", "perl", "c#"
    },
    "Web & Frontend": {
        "html", "css", "react", "angular", "vue", "nextjs", "redux",
        "bootstrap", "tailwind", "webpack", "sass", "scss", "jquery",
        "rest", "graphql", "api", "http", "json", "xml",
    },
    "Backend & Frameworks": {
        "django", "flask", "fastapi", "spring", "express", "nodejs",
        "laravel", "rails", "dotnet", "aspnet", "microservice", "serverless",
    },
    "Data & Machine Learning": {
        "machine", "learning", "deep", "neural", "nlp", "natural",
        "language", "processing", "tensorflow", "pytorch", "keras",
        "sklearn", "scikit", "pandas", "numpy", "matplotlib", "seaborn",
        "data", "analysis", "analytics", "statistic", "regression",
        "classification", "clustering", "model", "training", "inference",
    },
    "Databases": {
        "sql", "mysql", "postgresql", "postgres", "mongodb", "redis",
        "elasticsearch", "cassandra", "oracle", "sqlite", "nosql",
        "database", "query", "schema", "migration",
    },
    "Cloud & DevOps": {
        "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "k8s",
        "ci", "cd", "jenkins", "github", "gitlab", "terraform", "ansible",
        "linux", "unix", "devops", "deployment", "pipeline", "container",
    },
    "Soft Skills": {
        "communication", "leadership", "teamwork", "collaboration",
        "problem", "solving", "critical", "thinking", "management",
        "agile", "scrum", "kanban", "presentation", "mentoring",
    },
    "Tools & Productivity": {
        "git", "jira", "confluence", "slack", "trello", "notion",
        "figma", "postman", "vscode", "intellij", "excel", "powerpoint",
    },
}


def group_skills(cleaned_text: str) -> dict[str, list[str]]:
    """
    Match tokens in *cleaned_text* against the skill taxonomy.

    Parameters
    ----------
    cleaned_text : str
        Output of ``clean_text()`` — space-separated lemmatised tokens.

    Returns
    -------
    dict[str, list[str]]
        Mapping of category → sorted list of matched skill keywords.
        Empty categories are excluded.
    """
    if not cleaned_text.strip():
        return {}

    tokens = set(cleaned_text.split())
    grouped: dict[str, list[str]] = defaultdict(list)

    for category, keywords in SKILL_TAXONOMY.items():
        matched = sorted(tokens & keywords)
        if matched:
            grouped[category] = matched

    return dict(grouped)