/**
 * Test script to verify ML API integration
 * Run with: node test-ml-integration.js
 */

const { getGradingScores, checkMLAPIHealth } = require('./src/services/gradingService');
const path = require('path');

async function testMLIntegration() {
    console.log('🔍 Testing ML API Integration...\n');
    
    // 1. Test ML API Health
    console.log('1. Checking ML API Health...');
    try {
        const health = await checkMLAPIHealth();
        console.log('   Health Status:', health.status);
        if (health.status === 'healthy') {
            console.log('   ✅ ML API is running and healthy');
            if (health.data) {
                console.log('   API Response:', health.data);
            }
        } else {
            console.log('   ❌ ML API is not healthy:', health.error);
            console.log('   💡 Make sure to run the ML services first:');
            console.log('      cd ML && python unified_grading_api.py');
        }
    } catch (error) {
        console.log('   ❌ Health check failed:', error.message);
    }
    
    console.log('\n' + '='.repeat(50) + '\n');
    
    // 2. Test Grading Service (if we have a test file)
    console.log('2. Testing Grading Service...');
    
    // Look for any file in the uploads directory for testing
    const fs = require('fs');
    const uploadsDir = path.join(__dirname, 'public/uploads/homeworks');
    
    if (!fs.existsSync(uploadsDir)) {
        console.log('   ⚠️  No uploads directory found. Skipping file grading test.');
        console.log('   📁 Expected directory:', uploadsDir);
        return;
    }
    
    const files = fs.readdirSync(uploadsDir);
    const testFiles = files.filter(f => f.endsWith('.pdf') || f.endsWith('.docx') || f.endsWith('.txt'));
    
    if (testFiles.length === 0) {
        console.log('   ⚠️  No test files found in uploads directory. Skipping file grading test.');
        console.log('   📁 Looking in:', uploadsDir);
        console.log('   💡 Upload a homework file first, then run this test.');
        return;
    }
    
    const testFile = path.join(uploadsDir, testFiles[0]);
    console.log('   📄 Testing with file:', testFiles[0]);
    
    const studentInfo = {
        studentId: 'test_student_id',
        name: 'Test',
        lastname: 'Student'
    };
    
    const homeworkInfo = {
        homeworkId: 'test_homework_id',
        title: 'Test Assignment'
    };
    
    const classroomInfo = {
        name: 'Test Classroom'
    };
    
    try {
        console.log('   🔄 Calling ML grading service...');
        const startTime = Date.now();
        
        const results = await getGradingScores(testFile, studentInfo, homeworkInfo, classroomInfo);
        
        const duration = Date.now() - startTime;
        console.log(`   ⏱️  Analysis completed in ${duration}ms`);
        
        console.log('\n   📊 Grading Results:');
        console.log(`   • Similarity Score: ${results.similarityScore}%`);
        console.log(`   • AI Generated Score: ${results.aiGeneratedScore}%`);
        console.log(`   • Plagiarism Score: ${results.plagiarismScore}%`);
        
        if (results.reportHtml) {
            console.log(`   • Report HTML: ${results.reportHtml.length} characters`);
        }
        
        if (results.reportPath) {
            console.log(`   • Report Path: ${results.reportPath}`);
        }
        
        if (results.error) {
            console.log(`   ⚠️  Warning: ${results.error}`);
        }
        
        console.log('\n   ✅ Grading service test completed successfully!');
        
        // Show sample of report HTML
        if (results.reportHtml && results.reportHtml.length > 0) {
            const preview = results.reportHtml.substring(0, 200) + '...';
            console.log('\n   📄 Report Preview:');
            console.log('   ' + preview);
        }
        
    } catch (error) {
        console.log('   ❌ Grading test failed:', error.message);
        console.log('   💡 Error details:', error);
    }
}

// Run the test
if (require.main === module) {
    testMLIntegration()
        .then(() => {
            console.log('\n🎉 ML Integration test completed!');
            console.log('\n💡 Tips:');
            console.log('   • Make sure ML services are running: cd ML && python unified_grading_api.py');
            console.log('   • Check that all Python dependencies are installed');
            console.log('   • Upload some homework files to test with real content');
            process.exit(0);
        })
        .catch((error) => {
            console.error('\n💥 Test failed:', error);
            process.exit(1);
        });
}

module.exports = { testMLIntegration };
