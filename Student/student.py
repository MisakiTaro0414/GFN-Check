import xml.etree.ElementTree as ET
import xmlschema
import random
import json

with open('Student/config.json', 'r') as file:
    config = json.load(file)
    tag_options = config["tags"]
    text_options = config["texts"]
    num_children_options = list(range(5))


class Student:
    def __init__(self, xml):
        self.schema = xmlschema.XMLSchema("Student/student.xsd")
        self.xml = xml

    def valid(self):
        try:
            self.schema.validate(self.xml)
            return True
        except:
            return False

    def indent(self, elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def __repr__(self):
        self.indent(self.xml)
        return ET.tostring(self.xml, encoding='unicode')


def generate_student(oracle, max_depth):
    def _generate_student(oracle, node, depth):
        # Tag part
        tag = None
        if node is None:
            tag = ET.Element('student')
        else:
            tag_name = oracle.select(1)
            tag = ET.Element(tag_name)
        # Children part
        if depth < max_depth:
            num_children = oracle.select(2)
            if num_children > 0:
                for _ in range(num_children):
                    child = _generate_student(oracle, tag, depth + 1)
                    if child is not None:
                        tag.append(child)
            else:
                # Add text
                text = random.choice(text_options)
                tag.text = text
        else:
            # Add text
            text = random.choice(text_options)
            tag.text = text
        return tag

    student = Student(_generate_student(oracle, None, 0))
    return student, 0, student.valid()
