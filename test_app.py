"""
DrainSight AI - Comprehensive Verification Test Suite
Tests all routes, sample inspections, database logging, and report generation.
"""

import os
import unittest
from app import app
import database as db

class DrainSightTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_01_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'DrainSight', response.data)
        self.assertIn(b'Intelligent Drainage Monitoring System', response.data)

    def test_02_static_pages(self):
        for route in ['/upload-image', '/upload-video', '/dashboard', '/how-it-works', '/about']:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with {response.status_code}")

    def test_03_analyze_samples(self):
        for sample in ['clear', 'partial', 'heavy']:
            response = self.client.post(f'/analyze-sample/{sample}', follow_redirects=True)
            self.assertEqual(response.status_code, 200, f"Sample {sample} analysis failed")
            self.assertIn(b'Inspection Diagnosis Report', response.data)

    def test_04_dashboard_and_stats(self):
        # API stats endpoint
        api_res = self.client.get('/api/stats')
        self.assertEqual(api_res.status_code, 200)
        data = api_res.get_json()
        self.assertIn('total_inspections', data)
        self.assertGreaterEqual(data['total_inspections'], 3)

        # Dashboard page
        dash_res = self.client.get('/dashboard')
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn(b'Recent Inspection History', dash_res.data)

    def test_05_download_report(self):
        res = self.client.get('/download-report')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'text/csv')
        self.assertIn(b'Analysis ID', res.data)
        self.assertIn(b'Blockage (%)', res.data)

if __name__ == '__main__':
    unittest.main()
