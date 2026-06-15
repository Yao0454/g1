import re
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _parse(path: Path) -> tuple[str, str, str]:
    """
    读一个 skill.md 文件 将文件按规定的格式拆成 (name, description, body) 三个部分

    Args:
        path: Path => 文件路径

    Returns:
        tuple[str, str, str] => (name, description, body)
    """
    text: str = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if m is None:
        raise RuntimeError(
            f"{path.name} 缺少 frontmatter（格式应为 ---\\n...\\n---\\n正文）"
        )

    meta: dict = dict(re.findall(r"(\w+):\s*(.+)", m.group(1)))
    name: str = meta["name"]
    description: str = meta["description"]
    body: str = m.group(2).strip()
    return name, description, body


def list_skills() -> list[Path]:
    """
    列举出 skills 文件夹中现有的 skill

    Returns:
        list[Path] => skills 文件夹中现有的 skill 的路径
    """
    return list(SKILLS_DIR.glob("*.md"))


def list_skills_detail() -> list[tuple[str, str]]:
    """
    返回 skills 文件夹中现有的 skill 的 name 和 description

    Returns:
        list[tuple[str, str]] | [(name, desc) ,...]
                => name: skill 名称
                   desc: skill 概述
    """

    files: list[Path] = list_skills()
    skills: list[tuple[str, str, str]] = [_parse(file) for file in files]
    return [(name, desc) for name, desc, _ in skills]


def load_skill(name: str) -> str:
    """
    根据 name 找到对应的 skill 文件，返回正文

    Args:
        name: str => skill name

    Returns:
        str       => skill 的正文
               or => skill not found
    """

    for file in list_skills():
        skill_name, _, body = _parse(file)
        if skill_name == name:
            return body

    return f"{name} skill not found!"


if __name__ == "__main__":
    print(_parse(Path("/Volumes/1TKingston/g1/skills/greet.md")))
