@php
    $title = $gender === 1 ? 'Mr.' : ($gender === 0 ? 'Ms.' : '');
@endphp

<table width="100%" cellpadding="0" cellspacing="0" border="0"
       style="margin:0;padding:0;background:#f5f5f5;
              font-family:Cambria, Georgia, serif;">
    <tr>
        <td align="center">

            <!-- MAIN CONTAINER -->
            <table width="600" cellpadding="0" cellspacing="0" border="0"
                   style="background:#ffffff;">

                <!-- HEADER -->
                <tr>
                    <td align="center"
                        style="background:#06134D;
                               padding:24px 0;">

                        <!-- LOGO -->
                        <img src="https://roodio.blob.core.windows.net/uploads/images/logo-with-text.png"
                             width="140"
                             style="display:block;margin:0 auto 12px auto;"
                             alt="ROODIO Logo">

                        <!-- TITLE -->
                        <div style="color:#FFD1A6;
                                    font-size:28px;
                                    font-weight:bold;
                                    letter-spacing:2px;">
                            TWO WAY VERIFICATION
                        </div>
                    </td>
                </tr>

                <!-- BODY -->
                <tr>
                    <td style="padding:28px;
                               border:1px dashed #142C80;
                               font-size:16px;
                               line-height:24px;
                               color:#000000;">

                        <!-- GREETING -->
                        <p style="margin:0 0 16px 0;">
                            Dear
                            @if($title)
                                <strong> {{ $title }}</strong>
                            @endif
                            <strong style="color:#FF8E2B;">
                                {{ Str::upper($fullname) }}
                            </strong>,
                        </p>

                        <!-- INTRO -->
                        <p style="margin:0 0 20px 0;">
                            You recently requested a
                            <strong style="color:#FF8E2B;">
                                One-Time Password (OTP)
                            </strong>
                            to continue with your authentication process.
                            Please use the code below to complete the verification.
                        </p>

                        <!-- OTP BOX -->
                        <table width="100%" cellpadding="0" cellspacing="0" border="0">
                            <tr>
                                <td align="center" style="padding:10px 0 20px 0;">

                                    <table cellpadding="0" cellspacing="0" border="0">
                                        <tr>
                                            <td align="center"
                                                style="border:3px solid #FFC48D;
                                                       border-radius:8px;
                                                       font-size:42px;
                                                       letter-spacing:12px;
                                                       font-weight:bold;
                                                       padding:12px 48px 12px 60px;
                                                       color:#1F3A98;">
                                                {{ $otp }}
                                            </td>
                                        </tr>
                                    </table>

                                </td>
                            </tr>
                        </table>

                        <!-- WARNING -->
                        <p style="margin:0 0 16px 0;">
                            This code is valid
                            <strong style="color:#FF8E2B;">
                                for 5 minutes only
                            </strong>.
                            For your security,
                            <strong style="color:#FF8E2B;">
                                please do not share this OTP with anyone
                            </strong>.
                            Our team will never ask for your OTP.
                        </p>

                        <!-- IGNORE MESSAGE -->
                        <p style="margin:0 0 24px 0;">
                            If you did not request this code, you can safely ignore this email.
                            No changes will be made to your account without verification.
                        </p>

                        <!-- SIGNATURE -->
                        <p style="margin:0;">Best Regards,</p>
                        <p style="margin:0 0 24px 0;">
                            <strong>ROODIO Team</strong>
                        </p>

                        <hr style="border:0;border-top:1px dashed #808080;margin:20px 0;">

                        <!-- CONTACT -->
                        <p style="text-align:center;margin:0 0 8px 0;">
                            <strong>Our Contact</strong>
                        </p>

                        <p style="text-align:center;margin:0 0 20px 0;">
                            <a href="mailto:roodio.team@roodio.id"
                            style="color:#1a0dab;text-decoration:underline;"
                            target="_blank">
                                roodio.team@gmail.com
                            </a>
                        </p>


                        <hr style="border:0;border-top:1px dashed #808080;margin:20px 0;">

                        <!-- FOOTER NOTE -->
                        <p style="text-align:center;
                                  color:#808080;
                                  margin:0 0 5px 0;">
                            This email was sent automatically for security purposes.
                        </p>

                        <p style="text-align:center;
                                  color:#808080;
                                  margin:0;">
                            <strong>Please do not reply to this message.</strong>
                        </p>

                    </td>
                </tr>

            </table>

        </td>
    </tr>
</table>
