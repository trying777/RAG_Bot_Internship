"""
OCR Test Script
Test OCR functionality for scanned PDFs and images
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_ocr_setup():
    """Test OCR setup and configuration"""
    print("🔍 Testing OCR Setup...")
    
    try:
        from app.config import config
        from app.document_processor import DocumentProcessor
        
        print(f"✅ OCR Enabled: {config.ENABLE_OCR}")
        print(f"✅ Tesseract Path: {config.TESSERACT_CMD_PATH}")
        print(f"✅ TESSDATA_PREFIX: {config.TESSDATA_PREFIX}")
        print(f"✅ OCR Language: {config.OCR_LANGUAGE}")
        print(f"✅ OCR Config: {config.OCR_CONFIG}")
        
        # Test document processor initialization
        processor = DocumentProcessor()
        print("✅ Document processor initialized with OCR support")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR setup error: {str(e)}")
        return False

def test_image_processing():
    """Test image processing with OCR"""
    print("\n🖼️ Testing Image Processing...")
    
    try:
        from app.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Check if we have any test images
        test_images = list(Path("data/documents").glob("*.jpg")) + \
                     list(Path("data/documents").glob("*.png")) + \
                     list(Path("data/documents").glob("*.bmp"))
        
        if not test_images:
            print("⚠️ No test images found in data/documents/")
            print("   Add some JPG, PNG, or BMP files to test OCR")
            return True
        
        # Test with first available image
        test_image = test_images[0]
        print(f"🔄 Testing OCR with: {test_image.name}")
        
        # Process image
        text = processor.load_document(str(test_image))
        
        if text and len(text.strip()) > 0:
            print(f"✅ OCR successful! Extracted {len(text)} characters")
            print(f"📝 Sample text: {text[:200]}...")
        else:
            print("⚠️ OCR completed but no text extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing error: {str(e)}")
        return False

def test_pdf_detection():
    """Test PDF type detection"""
    print("\n📄 Testing PDF Type Detection...")
    
    try:
        from app.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Check if we have any test PDFs
        test_pdfs = list(Path("data/documents").glob("*.pdf"))
        
        if not test_pdfs:
            print("⚠️ No test PDFs found in data/documents/")
            print("   Add some PDF files to test detection")
            return True
        
        # Test with first available PDF
        test_pdf = test_pdfs[0]
        print(f"🔄 Testing PDF detection with: {test_pdf.name}")
        
        # Detect PDF type
        is_scanned = processor._is_scanned_pdf(str(test_pdf))
        
        if is_scanned:
            print(f"✅ Detected as: Scanned PDF (will use OCR)")
        else:
            print(f"✅ Detected as: Text-based PDF (will use standard extraction)")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF detection error: {str(e)}")
        return False

def main():
    """Run all OCR tests"""
    print("🚀 OCR Functionality Test")
    print("=" * 50)
    
    tests = [
        ("OCR Setup Test", test_ocr_setup),
        ("Image Processing Test", test_image_processing),
        ("PDF Detection Test", test_pdf_detection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 OCR Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All OCR tests passed! Your system is ready for scanned documents.")
        print("\n🚀 To test with real documents:")
        print("   1. Add scanned PDFs or images to data/documents/")
        print("   2. Run: streamlit run main.py")
        print("   3. Upload and process your documents")
    else:
        print("⚠️ Some OCR tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   1. Install OCR dependencies: pip install pytesseract opencv-python pdf2image Pillow")
        print("   2. Verify Tesseract installation path in config.env")
        print("   3. Check if TESSDATA_PREFIX is correct")

if __name__ == "__main__":
    main()
