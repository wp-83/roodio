<?php
namespace Tests\Feature\Auth;

use App\Http\Controllers\OtpController;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Mockery\MockInterface;
use Tests\TestCase;

class RegisterOtpTest extends TestCase
{
    use RefreshDatabase;

    public function test_verifikasi_otp_berhasil()
    {
        $this->mock(OtpController::class, function (MockInterface $mock) {
            $mock->shouldReceive('verify')->once()->andReturn(true);
        });

        $sessionData = [
            'fullname' => 'John',
            'email'    => 'test@example.com',
            // Data lain tidak wajib untuk step ini, yg penting email ada
        ];

        $response = $this->withSession([
            'register.step1'           => $sessionData,
            'user_verification_passed' => true, // Sesuaikan flag dari Step 1
        ])
            ->from(route('register.validation')) // PENTING: Asal request
            ->post(route('register.validation'), [
                'otp-1' => '1', 'otp-2' => '2', 'otp-3' => '3',
                'otp-4' => '4', 'otp-5' => '5', 'otp-6' => '6',
            ]);

        $response->assertRedirect(route('account'));
        $response->assertSessionHas('otp_passed', true);
    }

    public function test_verifikasi_otp_gagal_jika_kode_salah()
    {
        $this->mock(OtpController::class, function (MockInterface $mock) {
            $mock->shouldReceive('verify')->once()->andReturn(false);
        });

        $sessionData = ['email' => 'test@example.com'];

        $response = $this->withSession([
            'register.step1'           => $sessionData,
            'user_verification_passed' => true,
        ])
            ->from(route('register.validation')) // PENTING: Agar redirect back() benar
            ->post(route('register.validation'), [
                'otp-1' => '0', 'otp-2' => '0', 'otp-3' => '0',
                'otp-4' => '0', 'otp-5' => '0', 'otp-6' => '0',
            ]);

        // Assert: Redirect kembali ke halaman validasi (OTP)
        $response->assertRedirect(route('register.validation'));
        $response->assertSessionHasErrors(['otp']);
    }
}
