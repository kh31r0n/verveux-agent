
from typing import List, Dict, Any

class KnowledgeBuilder:
    def __init__(self, knowledge: List[Dict[str, Any]]):
        self.knowledge = knowledge

    def get_faqs_block(self) -> str:
        faqs = [item for item in self.knowledge if item.get("type") == "faq"]
        if not faqs:
            return ""

        faq_lines = []
        for faq in faqs:
            faq_lines.append(f"- {faq.get('question')}: {faq.get('answer')}")
        return "\n".join(faq_lines)

    def get_catalog_block(self) -> str:
        catalog_items = [item for item in self.knowledge if item.get("type") == "catalog_item"]
        if not catalog_items:
            return ""

        catalog_lines = []
        for item in catalog_items:
            catalog_lines.append(f"- {item.get('name')}: ${item.get('price', 0):.2f} (stock: {item.get('stock', 'N/A')})")
        return "\n".join(catalog_lines)

    def get_full_block(self) -> str:
        return f"""**FAQs**
{self.get_faqs_block()}

**Catálogo**
{self.get_catalog_block()}"""
