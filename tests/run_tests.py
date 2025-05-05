#!/usr/bin/env python3
"""
Test Runner for Lucky Train AI Assistant

This script discovers and runs all tests in the tests directory,
and generates a report of the results.
"""

import os
import sys
import unittest
import argparse
import time
from datetime import datetime
import json
import xmlrunner

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests(output_dir=None, xml_report=False, verbose=False):
    """Run all tests in the tests directory.
    
    Args:
        output_dir: Directory to save test reports
        xml_report: Whether to generate XML reports
        verbose: Whether to show verbose output
    
    Returns:
        Test result object
    """
    # Discover tests
    start_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add discovered tests
    discovered_suite = loader.discover(start_dir, pattern="test_*.py")
    suite.addTest(discovered_suite)
    
    # Add specific tests that might not be discovered automatically
    specific_modules = [
        'test_api',
        'test_vector_db',
        'test_jwt_auth'
    ]
    
    for module_name in specific_modules:
        try:
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
            module = __import__(module_name)
            module_suite = loader.loadTestsFromModule(module)
            suite.addTest(module_suite)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load tests from {module_name}: {e}")
    
    # Run tests
    start_time = time.time()
    
    if xml_report and output_dir:
        runner = xmlrunner.XMLTestRunner(output=output_dir, verbosity=2 if verbose else 1)
    else:
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    
    result = runner.run(suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Generate report
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": result.testsRun,
            "errors": len(result.errors),
            "failures": len(result.failures),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
            "success": result.wasSuccessful(),
            "duration": duration
        }
        
        # Add details about failures and errors
        if result.failures:
            report["failure_details"] = []
            for test, traceback in result.failures:
                report["failure_details"].append({
                    "test": str(test),
                    "traceback": traceback
                })
        
        if result.errors:
            report["error_details"] = []
            for test, traceback in result.errors:
                report["error_details"].append({
                    "test": str(test),
                    "traceback": traceback
                })
        
        # Write JSON report
        json_report_path = os.path.join(output_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Report saved to {json_report_path}")
    
    return result

def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run tests for Lucky Train AI Assistant')
    parser.add_argument('--output-dir', '-o', help='Directory to save test reports')
    parser.add_argument('--xml', '-x', action='store_true', help='Generate XML reports')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.xml and not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(__file__), '..', 'test-reports')
    
    # Run tests
    print("Running tests for Lucky Train AI Assistant...")
    print("=" * 70)
    
    result = run_tests(args.output_dir, args.xml, args.verbose)
    
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Success: {result.wasSuccessful()}")
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(main()) 