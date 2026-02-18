"""Test SkillLibrary save/load/search."""
import sys
sys.path.insert(0, r"c:\Users\gabri\JadeAgent")
import tempfile

from jadeagent.skills import SkillLibrary, SkillGenerator
print("Skills imports OK")

# Create temp library
tmp = tempfile.mkdtemp()
lib = SkillLibrary(tmp)

# Save a skill
code = '''def reverse_text(text: str) -> str:
    """Reverse the input text."""
    return text[::-1]
'''
lib.save("reverse_text", code, "Reverse a string")
print("Saved:", lib.list_skills())

# Search
tools = lib.search("reverse")
print("Found:", [t.name for t in tools])

# Execute
result = tools[0].execute({"text": "hello"})
print("Result:", result)
assert result == "olleh", f"Expected 'olleh', got '{result}'"

# Test persistence — reload from disk
lib2 = SkillLibrary(tmp)
tools2 = lib2.search("reverse")
print("Persist reload:", [t.name for t in tools2])
result2 = tools2[0].execute({"text": "JadeAgent"})
print("Persist result:", result2)
assert result2 == "tnegAedaJ"

print("\nALL TESTS PASSED!")
